// DiTBench.swift — Benchmark DiT ops (f32 vs f16), conv3d, flash attn

import XCTest
import Foundation
import Metal
import FlashAttention
@testable import SwiftLLM

// MARK: - Timing

private let ctx = MetalContext.shared

private func timeBatch(_ body: () -> Void) -> Double {
    let drain = MetalContext.shared.commandQueue.makeCommandBuffer()!
    drain.commit(); drain.waitUntilCompleted()
    ctx.beginBatch()
    let t0 = CFAbsoluteTimeGetCurrent()
    body()
    ctx.endBatch()
    return (CFAbsoluteTimeGetCurrent() - t0) * 1000.0
}

private func fmt(_ ms: Double) -> String { String(format: "%8.1f", ms) }
private func pct(_ a: Double, _ b: Double) -> String { String(format: "%.0f%%", b / a * 100) }
private func tStr(_ M: Int, _ K: Int, _ N: Int, ms: Double) -> String {
    String(format: "%.2f TFLOPS", Double(2*M*K*N) / (ms*1e-3) / 1e12)
}

// MARK: - Buffer helpers

private func f16buf(_ n: Int) -> MTLBuffer {
    let b = MetalContext.shared.device.makeBuffer(length: max(n*2,4), options: .storageModeShared)!
    b.contents().assumingMemoryBound(to: UInt16.self).initialize(repeating: 0x211A, count: n)
    return b
}
private func f32buf(_ n: Int) -> MTLBuffer {
    let b = MetalContext.shared.device.makeBuffer(length: max(n*4,4), options: .storageModeShared)!
    b.contents().assumingMemoryBound(to: Float.self).initialize(repeating: 0.01, count: n)
    return b
}
private func f16t(_ n: Int, shape: [Int]) -> Tensor { Tensor(buffer: f16buf(n), shape: shape, dtype: .float16) }

// MARK: - Fake weights

private func linF16(K: Int, N: Int) -> DiTLinear {
    DiTLinear(weight: f16t(N*K, shape:[N,K]), bias: nil, K: K, N: N)
}
private func rmsW(dim: Int) -> DiTRMSNormWeights {
    let b = MetalContext.shared.device.makeBuffer(length: dim*4, options: .storageModeShared)!
    b.contents().assumingMemoryBound(to: Float.self).initialize(repeating: 1.0, count: dim)
    return DiTRMSNormWeights(weight: Tensor(buffer: b, shape: [dim], dtype: .float32))
}
private func makeBlk(dim: Int, inner: Int, numAda: Int) -> DiTBlockWeights {
    DiTBlockWeights(
        norm1: rmsW(dim:dim),
        attn1: DiTAttentionWeights(toQ:linF16(K:dim,N:dim), toK:linF16(K:dim,N:dim),
                                    toV:linF16(K:dim,N:dim), toOut:linF16(K:dim,N:dim),
                                    qNorm:rmsW(dim:dim), kNorm:rmsW(dim:dim), useRope:true, isCrossAttention:false),
        norm2: rmsW(dim:dim),
        attn2: DiTAttentionWeights(toQ:linF16(K:dim,N:dim), toK:linF16(K:dim,N:dim),
                                    toV:linF16(K:dim,N:dim), toOut:linF16(K:dim,N:dim),
                                    qNorm:rmsW(dim:dim), kNorm:rmsW(dim:dim), useRope:false, isCrossAttention:true),
        ff: DiTFFNWeights(proj:linF16(K:dim,N:inner*2), projOut:linF16(K:inner,N:dim), isGeglu:true),
        scaleShiftTable: f16t(numAda*dim, shape:[numAda,dim]), numAdaParams: numAda)
}

// MARK: - f16 flash attention

private func flashAttnF16(q: Tensor, k: Tensor, v: Tensor, R: Int, C: Int, D: Int, H: Int) -> Tensor {
    var desc = AttentionDescriptor()
    desc.lowPrecisionInputs = true; desc.lowPrecisionIntermediates = true; desc.lowPrecisionOutputs = true
    desc.matrixDimensions = (row: UInt32(R), column: UInt32(C), head: UInt16(D))
    desc.transposeState = (Q:false, K:false, V:false, O:false); desc.causal = false
    let (kernel, pipeline) = AttentionKernel.pipeline(for: desc, type: .forward)
    let pool = MetalContext.shared.bufferPool
    let bufO = pool.get(length: H*R*D*2); let bufL = pool.get(length: H*R*4); let dummy = pool.get(length: 16)
    memset(bufO.contents(), 0, H*R*D*2); memset(bufL.contents(), 0, H*R*4); memset(dummy.contents(), 0, 16)
    let d = UInt32(D)
    var p: [UInt32] = [UInt32(H),1, UInt32(R)*d,UInt32(C)*d,UInt32(C)*d, UInt32(R)*d,UInt32(R),UInt32(R),
                       UInt32(R)*d,UInt32(C)*d,UInt32(C)*d,UInt32(R)*d, 0,UInt32(R),UInt32(C)]
    let bp = MetalContext.shared.device.makeBuffer(bytes:&p, length:p.count*4, options:.storageModeShared)!
    MetalContext.shared.run { enc in
        enc.setBuffer(q.buffer,offset:0,index:0); enc.setBuffer(k.buffer,offset:0,index:1)
        enc.setBuffer(v.buffer,offset:0,index:2); enc.setBuffer(bufO,offset:0,index:3)
        enc.setBuffer(bufL,offset:0,index:4)
        enc.setBuffer(dummy,offset:0,index:5); enc.setBuffer(dummy,offset:0,index:6)
        enc.setBuffer(dummy,offset:0,index:7); enc.setBuffer(dummy,offset:0,index:8)
        enc.setBuffer(dummy,offset:0,index:9)
        AttentionKernel.dispatch(encoder:enc, kernel:kernel, pipeline:pipeline, batchedParams:bp,
                                  parallelizationDimension:R, numHeads:H, batchSize:1)
    }
    return Tensor(buffer:bufO, shape:[H,R,D], dtype:.float16)
}

// MARK: - Block timing helpers

private struct BlockTimes {
    let norm1, q, k, v, attn, out, ff1, geglu, ff2: Double
    var total: Double { norm1+q+k+v+attn+out+ff1+geglu+ff2 }
}

@_optimize(none)
private func timeBlockF32(M: Int, dim: Int, nH: Int, hD: Int, inner: Int) -> BlockTimes {
    let block = makeBlk(dim:dim, inner:inner, numAda:6)
    let hid   = f32buf(M*dim)
    let pipe  = KernelCache.shared.pipeline("geglu_f16")
    let half  = M * inner

    let tN1  = timeBatch { let _ = block.norm1.applyF32(hid, dim:dim, rows:M) }
    let norm = block.norm1.applyF32(hid, dim:dim, rows:M)
    let tQ   = timeBatch { let _ = block.attn1.toQ.applyF32in(norm, M:M) }
    let tK   = timeBatch { let _ = block.attn1.toK.applyF32in(norm, M:M) }
    let tV   = timeBatch { let _ = block.attn1.toV.applyF32in(norm, M:M) }
    let qT   = transposeSHF32(block.attn1.toQ.applyF32in(norm,M:M), seqLen:M, nHeads:nH, headDim:hD)
    let kT   = transposeSHF32(block.attn1.toK.applyF32in(norm,M:M), seqLen:M, nHeads:nH, headDim:hD)
    let vT   = transposeSHF32(block.attn1.toV.applyF32in(norm,M:M), seqLen:M, nHeads:nH, headDim:hD)
    let tSA  = timeBatch { let _ = flashAttentionBidirectionalF32(q:qT,k:kT,v:vT,R:M,C:M,D:hD,nHeads:nH) }
    let aFlat = transposeHSF32(flashAttentionBidirectionalF32(q:qT,k:kT,v:vT,R:M,C:M,D:hD,nHeads:nH), seqLen:M, nHeads:nH, headDim:hD)
    let tOut = timeBatch { let _ = block.attn1.toOut.applyF32in(aFlat, M:M) }
    let normFF = block.norm2.applyF32(hid, dim:dim, rows:M)
    let tFF1 = timeBatch { let _ = block.ff.proj.applyF32in(normFF, M:M) }
    let p1   = block.ff.proj.applyF32in(normFF, M:M)
    let tGE  = timeBatch {
        let p16 = castF32toF16(p1, count:half*2, shape:[M,inner*2])
        let out = Tensor.empty([M,inner], dtype:.float16)
        ctx.run { enc in
            enc.setComputePipelineState(pipe)
            enc.setBuffer(p16.buffer,offset:half*2,index:0); enc.setBuffer(p16.buffer,offset:0,index:1)
            enc.setBuffer(out.buffer,offset:0,index:2)
            enc.dispatchThreads(MTLSize(width:half,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
        }
        let _ = castF16toF32(out)
    }
    let p16b = castF32toF16(p1, count:half*2, shape:[M,inner*2])
    let gb   = Tensor.empty([M,inner], dtype:.float16)
    ctx.beginBatch(); ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(p16b.buffer,offset:half*2,index:0); enc.setBuffer(p16b.buffer,offset:0,index:1)
        enc.setBuffer(gb.buffer,offset:0,index:2)
        enc.dispatchThreads(MTLSize(width:half,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
    }; ctx.endBatch()
    let tFF2 = timeBatch { let _ = block.ff.projOut.applyF32in(castF16toF32(gb), M:M) }
    return BlockTimes(norm1:tN1, q:tQ, k:tK, v:tV, attn:tSA, out:tOut, ff1:tFF1, geglu:tGE, ff2:tFF2)
}

@_optimize(none)
private func timeBlockF16(M: Int, dim: Int, nH: Int, hD: Int, inner: Int) -> BlockTimes {
    let block = makeBlk(dim:dim, inner:inner, numAda:6)
    let wn   = f16t(dim, shape:[dim])
    let hid  = f16t(M*dim, shape:[M,dim])
    let pipe = KernelCache.shared.pipeline("geglu_f16")
    let half = M * inner

    let tN1  = timeBatch { let _ = rmsNorm(hid, weight:wn, eps:1e-6, dim:dim) }
    let norm = rmsNorm(hid, weight:wn, eps:1e-6, dim:dim)
    let tQ   = timeBatch { let _ = block.attn1.toQ.applyF16(norm, M:M) }
    let tK   = timeBatch { let _ = block.attn1.toK.applyF16(norm, M:M) }
    let tV   = timeBatch { let _ = block.attn1.toV.applyF16(norm, M:M) }
    let qT   = transposeSH(block.attn1.toQ.applyF16(norm,M:M), seqLen:M, nHeads:nH, headDim:hD)
    let kT   = transposeSH(block.attn1.toK.applyF16(norm,M:M), seqLen:M, nHeads:nH, headDim:hD)
    let vT   = transposeSH(block.attn1.toV.applyF16(norm,M:M), seqLen:M, nHeads:nH, headDim:hD)
    let tSA  = timeBatch { let _ = flashAttnF16(q:qT,k:kT,v:vT,R:M,C:M,D:hD,H:nH) }
    let aFlat = transposeHS(flashAttnF16(q:qT,k:kT,v:vT,R:M,C:M,D:hD,H:nH), seqLen:M, nHeads:nH, headDim:hD)
    let tOut = timeBatch { let _ = block.attn1.toOut.applyF16(aFlat, M:M) }
    let normFF = rmsNorm(hid, weight:wn, eps:1e-6, dim:dim)
    let tFF1 = timeBatch { let _ = block.ff.proj.applyF16(normFF, M:M) }
    let p1   = block.ff.proj.applyF16(normFF, M:M)
    let tGE  = timeBatch {
        let out = Tensor.empty([M,inner], dtype:.float16)
        ctx.run { enc in
            enc.setComputePipelineState(pipe)
            enc.setBuffer(p1.buffer,offset:half*2,index:0); enc.setBuffer(p1.buffer,offset:0,index:1)
            enc.setBuffer(out.buffer,offset:0,index:2)
            enc.dispatchThreads(MTLSize(width:half,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
        }
    }
    let go   = Tensor.empty([M,inner], dtype:.float16)
    ctx.beginBatch(); ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(p1.buffer,offset:half*2,index:0); enc.setBuffer(p1.buffer,offset:0,index:1)
        enc.setBuffer(go.buffer,offset:0,index:2)
        enc.dispatchThreads(MTLSize(width:half,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
    }; ctx.endBatch()
    let tFF2 = timeBatch { let _ = block.ff.projOut.applyF16(go, M:M) }
    return BlockTimes(norm1:tN1, q:tQ, k:tK, v:tV, attn:tSA, out:tOut, ff1:tFF1, geglu:tGE, ff2:tFF2)
}

// MARK: - Benchmark class

final class DiTBench: XCTestCase {

    @_optimize(none)
    func testGEMM() {
        print("\n=== GEMM: f32-in×f16w→f32  vs  f16×f16→f16 ===")
        print(String(format: "%-32@ %8@ %8@ %7@  %@", "shape", "f32(ms)", "f16(ms)", "speedup", "f16 TFLOPS"))
        let cases: [(M:Int, K:Int, N:Int, String)] = [
            (4096, 2048, 2048,   "QKV/toOut [4k]"),
            (4096, 2048, 16384,  "FFN proj1 [4k]"),
            (4096, 8192, 2048,   "FFN proj2 [4k]"),
            (36864, 2048, 2048,  "QKV/toOut [36k]"),
            (36864, 2048, 16384, "FFN proj1 [36k]"),
            (36864, 8192, 2048,  "FFN proj2 [36k]"),
        ]
        for (M,K,N,label) in cases {
            let lin  = linF16(K:K, N:N)
            let wF16 = f16t(N*K, shape:[N,K])
            let xF32 = f32buf(M*K); let xF16 = f16t(M*K, shape:[M,K])
            // warm
            let _ = timeBatch { let _ = lin.applyF32in(xF32, M:M) }
            let _ = timeBatch { let _ = matmulF16(xF16, wF16, M:M, K:K, N:N) }
            let tF32 = timeBatch { let _ = lin.applyF32in(xF32, M:M) }
            let tF16 = timeBatch { let _ = matmulF16(xF16, wF16, M:M, K:K, N:N) }
            print(String(format: "%-32@ %8.1f %8.1f %7@  %@", label, tF32, tF16, pct(tF32,tF16), tStr(M,K,N,ms:tF16)))
        }
    }

    @_optimize(none)
    func testFlashAttention() {
        print("\n=== Flash Attention: f32  vs  f16 ===")
        print(String(format: "%-28@ %8@ %8@ %7@", "shape", "f32(ms)", "f16(ms)", "speedup"))
        let cases: [(R:Int, C:Int, H:Int, D:Int, String)] = [
            (4096,  4096, 32, 64, "self [4k×4k]"),
            (36864, 36864,32, 64, "self [36k×36k]"),
            (4096,  128,  32, 64, "cross [4k×128]"),
        ]
        for (R,C,H,D,label) in cases {
            let qF32=f32buf(H*R*D); let kF32=f32buf(H*C*D); let vF32=f32buf(H*C*D)
            let qF16=f16t(H*R*D,shape:[H,R,D]); let kF16=f16t(H*C*D,shape:[H,C,D]); let vF16=f16t(H*C*D,shape:[H,C,D])
            let _ = timeBatch { let _ = flashAttentionBidirectionalF32(q:qF32,k:kF32,v:vF32,R:R,C:C,D:D,nHeads:H) }
            let _ = timeBatch { let _ = flashAttnF16(q:qF16,k:kF16,v:vF16,R:R,C:C,D:D,H:H) }
            let tF32 = timeBatch { let _ = flashAttentionBidirectionalF32(q:qF32,k:kF32,v:vF32,R:R,C:C,D:D,nHeads:H) }
            let tF16 = timeBatch { let _ = flashAttnF16(q:qF16,k:kF16,v:vF16,R:R,C:C,D:D,H:H) }
            print(String(format: "%-28@ %8.1f %8.1f %7@", label, tF32, tF16, pct(tF32,tF16)))
            ctx.bufferPool.releaseAll(keeping:[]); ctx.bufferPool.trimFreeList()
        }
    }

    @_optimize(none)
    func testGEGLU() {
        print("\n=== GEGLU: f32 roundtrip  vs  pure f16 ===")
        print(String(format: "%-22@ %10@ %8@ %7@", "shape", "f32rt(ms)", "f16(ms)", "speedup"))
        let pipe = KernelCache.shared.pipeline("geglu_f16")
        for (M, inner, label) in [(4096, 8192, "GEGLU [4k]"), (36864, 8192, "GEGLU [36k]")] {
            let half = M * inner
            let p32 = f32buf(half*2); let p16 = f16t(half*2, shape:[M,inner*2])
            let tRT = timeBatch {
                let cast = castF32toF16(p32, count:half*2, shape:[M,inner*2])
                let out  = Tensor.empty([M,inner], dtype:.float16)
                ctx.run { enc in
                    enc.setComputePipelineState(pipe)
                    enc.setBuffer(cast.buffer,offset:half*2,index:0); enc.setBuffer(cast.buffer,offset:0,index:1)
                    enc.setBuffer(out.buffer,offset:0,index:2)
                    enc.dispatchThreads(MTLSize(width:half,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
                }
                let _ = castF16toF32(out)
            }
            let tF16 = timeBatch {
                let out = Tensor.empty([M,inner], dtype:.float16)
                ctx.run { enc in
                    enc.setComputePipelineState(pipe)
                    enc.setBuffer(p16.buffer,offset:half*2,index:0); enc.setBuffer(p16.buffer,offset:0,index:1)
                    enc.setBuffer(out.buffer,offset:0,index:2)
                    enc.dispatchThreads(MTLSize(width:half,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
                }
            }
            print(String(format: "%-22@ %10.1f %8.1f %7@", label, tRT, tF16, pct(tRT,tF16)))
        }
    }

    @_optimize(none)
    func testConv3d() {
        print("\n=== Conv3d upsampler: f32  vs  f16 ===")
        print(String(format: "%-38@ %8@ %8@ %7@", "shape", "f32(ms)", "f16(ms)", "speedup"))
        let cases: [(B:Int,Ci:Int,Co:Int,D:Int,H:Int,W:Int,kD:Int,kH:Int,kW:Int,pad:Int,String)] = [
            (1,128,512,16,16,16,1,1,1,0,"init_conv 128→512 [16³]"),
            (1,512,512,16,16,16,3,3,3,1,"resblock 512ch 3x3x3 [16³]"),
            (1,512,512,16,32,32,3,3,3,1,"resblock 512ch 3x3x3 [16×32²]"),
            (1,128,128,16,32,32,3,3,3,1,"resblock 128ch 3x3x3 [16×32²]"),
            (1,128,512,16,16,16,3,3,3,1,"resblock 128→512 3x3x3 [16³]"),
        ]
        for (B,Ci,Co,D,H,W,kD,kH,kW,pad,label) in cases {
            let xF32=f32buf(B*Ci*D*H*W); let wF32=f32buf(Co*Ci*kD*kH*kW); let bF32=f32buf(Co)
            let xF16=f16t(B*Ci*D*H*W,shape:[B,Ci,D,H,W]); let wF16=f16t(Co*Ci*kD*kH*kW,shape:[Co,Ci,kD,kH,kW]); let bF16=f16t(Co,shape:[Co])
            let _ = timeBatch { let _ = conv3dF32(xF32,weight:wF32,bias:bF32,B:B,C_in:Ci,D:D,H:H,W:W,C_out:Co,kD:kD,kH:kH,kW:kW,pD:pad,pH:pad,pW:pad) }
            let _ = timeBatch { let _ = conv3d(xF16,weight:wF16,bias:bF16,B:B,C_in:Ci,D:D,H:H,W:W,C_out:Co,kD:kD,kH:kH,kW:kW,pD:pad,pH:pad,pW:pad) }
            let tF32 = timeBatch { let _ = conv3dF32(xF32,weight:wF32,bias:bF32,B:B,C_in:Ci,D:D,H:H,W:W,C_out:Co,kD:kD,kH:kH,kW:kW,pD:pad,pH:pad,pW:pad) }
            let tF16 = timeBatch { let _ = conv3d(xF16,weight:wF16,bias:bF16,B:B,C_in:Ci,D:D,H:H,W:W,C_out:Co,kD:kD,kH:kH,kW:kW,pD:pad,pH:pad,pW:pad) }
            print(String(format: "%-38@ %8.1f %8.1f %7@", label, tF32, tF16, pct(tF32,tF16)))
            ctx.bufferPool.releaseAll(keeping:[]); ctx.bufferPool.trimFreeList()
        }
    }

    @_optimize(none)
    func testDiTBlock() {
        print("\n=== DiT block f32 vs f16 (9 ops timed) ===")
        let dim=2048, nH=32, hD=64, inner=8192
        let ops = ["norm1","self_Q","self_K","self_V","flashAttn","toOut","ffn_proj1","geglu","ffn_proj2","TOTAL"]
        for M in [4096] {  // 36k skipped: FFN buffers would be >1GB
            print("\n--- numTokens=\(M) ---")
            print(String(format: "  %-18@ %8@ %8@ %7@", "op", "f32(ms)", "f16(ms)", "speedup"))
            for run in 1...2 {
                let r32 = timeBlockF32(M:M, dim:dim, nH:nH, hD:hD, inner:inner)
                let r16 = timeBlockF16(M:M, dim:dim, nH:nH, hD:hD, inner:inner)
                let t32 = [r32.norm1,r32.q,r32.k,r32.v,r32.attn,r32.out,r32.ff1,r32.geglu,r32.ff2,r32.total]
                let t16 = [r16.norm1,r16.q,r16.k,r16.v,r16.attn,r16.out,r16.ff1,r16.geglu,r16.ff2,r16.total]
                print("  [Run \(run)]")
                for i in 0..<ops.count {
                    print(String(format: "  %-18@ %8.1f %8.1f %7@", ops[i], t32[i], t16[i], pct(t32[i],t16[i])))
                }
                ctx.bufferPool.releaseAll(keeping:[]); ctx.bufferPool.trimFreeList()
            }
        }
    }
}
