// High-level ops built on Metal kernels + FlashAttention.

import Metal
import FlashAttention

// MARK: - Element-wise Ops

public func rmsNorm(_ x: Tensor, weight: Tensor, eps: Float, dim: Int) -> Tensor {
    let rows = x.count / dim
    let out = Tensor.empty([rows, dim], dtype: .float16)
    let pipe = KernelCache.shared.pipeline("rms_norm")

    var d = UInt32(dim)
    var e = eps

    let tgSize = min(256, dim)
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        enc.setBytes(&d, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    return out
}

public func silu(_ x: Tensor) -> Tensor {
    let out = Tensor.empty(x.shape, dtype: .float16)
    let pipe = KernelCache.shared.pipeline("silu")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: x.count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Fused SiLU(a) * b — saves a kernel launch and memory round-trip.
public func siluMul(_ a: Tensor, _ b: Tensor) -> Tensor {
    let out = Tensor.empty(a.shape, dtype: .float16)
    let pipe = KernelCache.shared.pipeline("silu_mul")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: a.count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

public func elemMul(_ a: Tensor, _ b: Tensor) -> Tensor {
    let out = Tensor.empty(a.shape, dtype: .float16)
    let pipe = KernelCache.shared.pipeline("mul")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: a.count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

public func elemAdd(_ a: Tensor, _ b: Tensor) -> Tensor {
    let out = Tensor.empty(a.shape, dtype: .float16)
    let pipe = KernelCache.shared.pipeline("add")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: a.count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

public func embedding(table: Tensor, tokenId: Int, dim: Int) -> Tensor {
    let out = Tensor.empty([dim], dtype: .float16)
    var tid = UInt32(tokenId)
    var d = UInt32(dim)
    let pipe = KernelCache.shared.pipeline("embedding")

    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(table.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&tid, length: 4, index: 2)
        enc.setBytes(&d, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: dim, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Transpose [seqLen, nHeads, headDim] → [nHeads, seqLen, headDim]
public func transposeSH(_ x: Tensor, seqLen: Int, nHeads: Int, headDim: Int) -> Tensor {
    let out = Tensor.empty([nHeads, seqLen, headDim], dtype: .float16)
    var sl = UInt32(seqLen), nh = UInt32(nHeads), hd = UInt32(headDim)
    let pipe = KernelCache.shared.pipeline("transpose_sh")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&sl, length: 4, index: 2)
        enc.setBytes(&nh, length: 4, index: 3)
        enc.setBytes(&hd, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: headDim, height: seqLen * nHeads, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1))
    }
    return out
}

/// Transpose [nHeads, seqLen, headDim] → [seqLen, nHeads, headDim]
public func transposeHS(_ x: Tensor, seqLen: Int, nHeads: Int, headDim: Int) -> Tensor {
    let out = Tensor.empty([seqLen, nHeads, headDim], dtype: .float16)
    var sl = UInt32(seqLen), nh = UInt32(nHeads), hd = UInt32(headDim)
    let pipe = KernelCache.shared.pipeline("transpose_hs")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&sl, length: 4, index: 2)
        enc.setBytes(&nh, length: 4, index: 3)
        enc.setBytes(&hd, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: headDim, height: nHeads * seqLen, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1))
    }
    return out
}

/// Batch embedding lookup: [numTokens] → [numTokens, dim]
public func embeddingBatch(table: Tensor, tokenIds: [Int], dim: Int) -> Tensor {
    let n = tokenIds.count
    let out = Tensor.empty([n, dim], dtype: .float16)
    var ids = tokenIds.map { UInt32($0) }
    let ctx = MetalContext.shared
    let idsBuf = ctx.makePooledBuffer(&ids, length: n * 4)
    var d = UInt32(dim)
    let pipe = KernelCache.shared.pipeline("embedding_batch")

    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(table.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBuffer(idsBuf, offset: 0, index: 2)
        enc.setBytes(&d, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: dim, height: n, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Batch quantized embedding lookup: [numTokens] → [numTokens, K]
public func embeddingQ4Batch(weight: Tensor, scales: Tensor, biases: Tensor, tokenIds: [Int], K: Int, groupSize: Int) -> Tensor {
    let n = tokenIds.count
    let out = Tensor.empty([n, K], dtype: .float16)
    var ids = tokenIds.map { UInt32($0) }
    let ctx = MetalContext.shared
    let idsBuf = ctx.makePooledBuffer(&ids, length: n * 4)
    var k = UInt32(K)
    var gs = UInt32(groupSize)
    let pipe = KernelCache.shared.pipeline("embedding_q4_batch")

    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(weight.buffer, offset: 0, index: 0)
        enc.setBuffer(scales.buffer, offset: 0, index: 1)
        enc.setBuffer(biases.buffer, offset: 0, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        enc.setBuffer(idsBuf, offset: 0, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        enc.setBytes(&gs, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: K, height: n, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

public func embeddingQ4(weight: Tensor, scales: Tensor, biases: Tensor, tokenId: Int, K: Int, groupSize: Int) -> Tensor {
    let out = Tensor.empty([K], dtype: .float16)
    var tid = UInt32(tokenId)
    var k = UInt32(K)
    var gs = UInt32(groupSize)
    let pipe = KernelCache.shared.pipeline("embedding_q4")

    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(weight.buffer, offset: 0, index: 0)
        enc.setBuffer(scales.buffer, offset: 0, index: 1)
        enc.setBuffer(biases.buffer, offset: 0, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        enc.setBytes(&tid, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        enc.setBytes(&gs, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: K, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

public func rope(_ x: Tensor, headDim: Int, seqLen: Int, startPos: Int, freqs: MTLBuffer) -> Tensor {
    let out = Tensor.empty(x.shape, dtype: .float16)

    var hd = UInt32(headDim)
    var p = UInt32(startPos)
    var sl = UInt32(seqLen)
    let pipe = KernelCache.shared.pipeline("rope")

    let nPairs = headDim / 2
    let nHeadSeq = x.count / headDim

    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)   // src
        enc.setBuffer(out.buffer, offset: 0, index: 1)  // dst
        enc.setBytes(&hd, length: 4, index: 2)
        enc.setBytes(&p, length: 4, index: 3)
        enc.setBuffer(freqs, offset: 0, index: 4)       // precomputed freqs
        enc.setBytes(&sl, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: nPairs, height: nHeadSeq, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(nPairs, 32), height: 1, depth: 1))
    }
    return out
}

/// Fused transpose [seqLen, nHeads, headDim] → [nHeads, seqLen, headDim] + RoPE.
public func transposeSHRope(_ x: Tensor, seqLen: Int, nHeads: Int, headDim: Int, startPos: Int, freqs: MTLBuffer) -> Tensor {
    let out = Tensor.empty([nHeads, seqLen, headDim], dtype: .float16)
    var hd = UInt32(headDim), sl = UInt32(seqLen), nh = UInt32(nHeads), sp = UInt32(startPos)
    let nPairs = headDim / 2
    let pipe = KernelCache.shared.pipeline("transpose_sh_rope")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&hd, length: 4, index: 2)
        enc.setBytes(&sl, length: 4, index: 3)
        enc.setBytes(&nh, length: 4, index: 4)
        enc.setBytes(&sp, length: 4, index: 5)
        enc.setBuffer(freqs, offset: 0, index: 6)
        enc.dispatchThreads(MTLSize(width: nPairs, height: seqLen * nHeads, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(nPairs, 32), height: 1, depth: 1))
    }
    return out
}

/// Fused residual add + RMS norm: computes residual = a + b, then norm = rms_norm(residual, weight).
/// Returns (residual, norm) — both are needed downstream.
public func residualRmsNorm(_ a: Tensor, _ b: Tensor, weight: Tensor, eps: Float, dim: Int) -> (residual: Tensor, normed: Tensor) {
    let rows = a.count / dim
    let residual = Tensor.empty([rows, dim], dtype: .float16)
    let normed = Tensor.empty([rows, dim], dtype: .float16)
    var d = UInt32(dim)
    var e = eps
    let pipe = KernelCache.shared.pipeline("residual_rms_norm")
    let tgSize = min(dim, 256)
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(weight.buffer, offset: 0, index: 2)
        enc.setBuffer(residual.buffer, offset: 0, index: 3)
        enc.setBuffer(normed.buffer, offset: 0, index: 4)
        enc.setBytes(&d, length: 4, index: 5)
        enc.setBytes(&e, length: 4, index: 6)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    return (residual, normed)
}

// MARK: - Quantized matmul (4-bit)

/// Quantized matmul: out = x @ W^T where W is 4-bit packed.
/// x: [M, K] f16. Returns [M, N] f16.
public func matmulQ4(_ x: Tensor, weight: Tensor, scales: Tensor, biases: Tensor, M: Int, K: Int, N: Int, groupSize: Int) -> Tensor {
    let out = Tensor.empty([M, N], dtype: .float16)
    var params: [UInt32] = [UInt32(K), UInt32(groupSize), UInt32(N), UInt32(M)]
    let pipe = KernelCache.shared.pipeline("matmul_q4")

    let rowsPerTG = 8  // 2 simdgroups × 4 rows
    let tgSize = 64    // 2 simdgroups × 32 lanes
    let totalTGs = (M * N + rowsPerTG - 1) / rowsPerTG
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: 0, index: 1)
        enc.setBuffer(scales.buffer, offset: 0, index: 2)
        enc.setBuffer(biases.buffer, offset: 0, index: 3)
        enc.setBuffer(out.buffer, offset: 0, index: 4)
        enc.setBytes(&params, length: params.count * 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: totalTGs, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    return out
}

// MARK: - Fast quantized GEMM (monolithic IR)

/// Fast quantized matmul via monolithic IR kernel.
/// A: [M, K] f16 (input activations). W: [N, K/8] uint32 (packed weights).
/// scales/biases: [N, K/groupSize] f16. Output: [M, N] f16.
public func matmulQ4Fast(
    _ x: Tensor, weight: Tensor, scales: Tensor, biases: Tensor,
    M: Int, K: Int, N: Int, groupSize: Int
) -> Tensor {
    var gemmDesc = GEMMDescriptor()
    gemmDesc.matrixDimensions = (M: UInt32(M), N: UInt32(N), K: UInt32(K))
    gemmDesc.memoryPrecisions = (A: .FP16, B: .FP16, C: .FP16)
    gemmDesc.transposeState = (A: false, B: true)
    gemmDesc.quantizedB = true
    gemmDesc.groupSize = UInt32(groupSize)

    let (kernel, pipeline) = GEMMKernel.pipeline(for: gemmDesc)
    let out = Tensor.empty([M, N], dtype: .float16)

    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipeline)
        enc.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)

        enc.setBuffer(x.buffer, offset: 0, index: 0)       // A
        enc.setBuffer(weight.buffer, offset: 0, index: 1)   // W (packed)
        enc.setBuffer(scales.buffer, offset: 0, index: 2)   // scales
        enc.setBuffer(biases.buffer, offset: 0, index: 3)   // biases
        enc.setBuffer(out.buffer, offset: 0, index: 4)      // C

        let gridSize = MTLSize(
            width: (N + Int(kernel.blockDimensions.N) - 1) / Int(kernel.blockDimensions.N),
            height: (M + Int(kernel.blockDimensions.M) - 1) / Int(kernel.blockDimensions.M),
            depth: 1)
        enc.dispatchThreadgroups(
            gridSize,
            threadsPerThreadgroup: MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1))
    }
    return out
}

/// Plain f16×f16 GEMM via FlashAttention's monolithic IR kernel.
/// A: [M, K] f16, B: [K, N] f16 (or B^T: [N, K] if transposeB=true). Output: [M, N] f16.
/// If `into` is provided, writes result there instead of allocating.
public func matmulF16(
    _ a: Tensor, _ b: Tensor,
    M: Int, K: Int, N: Int,
    transposeA: Bool = false, transposeB: Bool = true,
    into: Tensor? = nil
) -> Tensor {
    var gemmDesc = GEMMDescriptor()
    gemmDesc.matrixDimensions = (M: UInt32(M), N: UInt32(N), K: UInt32(K))
    gemmDesc.memoryPrecisions = (A: .FP16, B: .FP16, C: .FP16)
    gemmDesc.transposeState = (A: transposeA, B: transposeB)
    gemmDesc.quantizedB = false

    let (kernel, pipeline) = GEMMKernel.pipeline(for: gemmDesc)
    let out = into ?? Tensor.empty([M, N], dtype: .float16)

    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipeline)
        enc.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)

        enc.setBuffer(a.buffer, offset: 0, index: 0)   // A
        enc.setBuffer(b.buffer, offset: 0, index: 1)    // B
        enc.setBuffer(out.buffer, offset: 0, index: 2)   // C

        let gridSize = MTLSize(
            width: (N + Int(kernel.blockDimensions.N) - 1) / Int(kernel.blockDimensions.N),
            height: (M + Int(kernel.blockDimensions.M) - 1) / Int(kernel.blockDimensions.M),
            depth: 1)
        enc.dispatchThreadgroups(
            gridSize,
            threadsPerThreadgroup: MTLSize(width: Int(kernel.threadgroupSize), height: 1, depth: 1))
    }
    return out
}

/// Simple tiled GEMM — no per-shape compilation. C[M,N] = A[M,K] × B^T[N,K].
/// Slower than FlashAttention GEMM but compiles once for all shapes.
public func simpleMatmulF16(
    _ a: Tensor, _ b: Tensor,
    M: Int, K: Int, N: Int,
    into: Tensor? = nil
) -> Tensor {
    let out = into ?? Tensor.empty([M, N], dtype: .float16)
    let pipe = KernelCache.shared.pipeline("simple_gemm_f16")
    var m = UInt32(M), n = UInt32(N), k = UInt32(K)
    // smem: max(512 halfs=1024B load, 1024 floats=4096B store) = 4096 bytes
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        enc.setBytes(&m, length: 4, index: 3)
        enc.setBytes(&n, length: 4, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        enc.setThreadgroupMemoryLength(4096, index: 0)
        let gridW = (N + 31) / 32
        let gridH = (M + 31) / 32
        enc.dispatchThreadgroups(MTLSize(width: gridW, height: gridH, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    }
    return out
}

/// Unified linear: handles both float and quantized weights.
public func linear(_ x: Tensor, _ w: LinearWeight, M: Int, K: Int, N: Int, fast: Bool = false) -> Tensor {
    switch w {
    case .float(let weight):
        return matmulCPU(x, weight, M: M, K: K, N: N)
    case .quantized(let weight, let scales, let biases, let wK, let groupSize):
        if fast && M > 1 {
            return matmulQ4Fast(x, weight: weight, scales: scales, biases: biases, M: M, K: wK, N: N, groupSize: groupSize)
        } else {
            return matmulQ4(x, weight: weight, scales: scales, biases: biases, M: M, K: wK, N: N, groupSize: groupSize)
        }
    }
}

// MARK: - Sampling

/// Argmax over a tensor (CPU — fine for vocab-sized vector). Supports f16 and f32.
public func argmax(_ x: Tensor) -> Int {
    if x.dtype == .float16 {
        let ptr = x.buffer.contents().assumingMemoryBound(to: Float16.self)
        var best = 0
        var bestVal = Float(ptr[0])
        for i in 1..<x.count {
            let v = Float(ptr[i])
            if v > bestVal {
                bestVal = v
                best = i
            }
        }
        return best
    }
    let ptr = x.buffer.contents().assumingMemoryBound(to: Float.self)
    var best = 0
    var bestVal = ptr[0]
    for i in 1..<x.count {
        if ptr[i] > bestVal {
            bestVal = ptr[i]
            best = i
        }
    }
    return best
}

/// Temperature sampling with top-p (nucleus) filtering and repetition penalty.
/// Sample with Set-based repetition penalty (O(1) lookup instead of O(n) array scan).
public func sample(_ x: Tensor, temperature: Float = 0.7, topP: Float = 0.9, repetitionPenalty: Float = 1.1, seenTokens: Set<Int> = []) -> Int {
    let n = x.count
    var logits = [Float](repeating: 0, count: n)
    if x.dtype == .float16 {
        let ptr = x.buffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<n { logits[i] = Float(ptr[i]) }
    } else {
        let ptr = x.buffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<n { logits[i] = ptr[i] }
    }

    // Apply repetition penalty
    for tok in seenTokens where tok < n {
        if logits[tok] > 0 {
            logits[tok] /= repetitionPenalty
        } else {
            logits[tok] *= repetitionPenalty
        }
    }

    return sampleFromLogits(&logits, temperature: temperature, topP: topP)
}

public func sample(_ x: Tensor, temperature: Float = 0.7, topP: Float = 0.9, repetitionPenalty: Float = 1.1, previousTokens: [Int] = []) -> Int {
    return sample(x, temperature: temperature, topP: topP, repetitionPenalty: repetitionPenalty, seenTokens: Set(previousTokens))
}

// MARK: - VAE 3D Ops

/// 3D Convolution — im2col+GEMM for small C_in, gather-GEMM for large C_in.
/// Input: [B, C_in, D, H, W], Weight: [C_out, C_in, kD, kH, kW].
/// Output: [B, C_out, D_out, H_out, W_out].
public func conv3d(_ x: Tensor, weight: Tensor, bias: Tensor,
                   B: Int, C_in: Int, D: Int, H: Int, W: Int,
                   C_out: Int, kD: Int, kH: Int, kW: Int,
                   sD: Int = 1, sH: Int = 1, sW: Int = 1,
                   pD: Int = 0, pH: Int = 0, pW: Int = 0) -> Tensor {
    let D_out = (D + 2*pD - kD) / sD + 1
    let H_out = (H + 2*pH - kH) / sH + 1
    let W_out = (W + 2*pW - kW) / sW + 1

    let isPointwise = kD == 1 && kH == 1 && kW == 1 && sD == 1 && sH == 1 && sW == 1 && pD == 0 && pH == 0 && pW == 0

    if isPointwise {
        // 1x1x1: input-stationary with shared memory (already fast)
        let out = Tensor.empty([B, C_out, D_out, H_out, W_out], dtype: .float16)
        let spatialSize = D * H * W
        let tgSize = min(C_out, 256)
        let tileC = min(C_in, 1024)
        var params: [UInt32] = [UInt32(C_in), UInt32(spatialSize), UInt32(C_out), UInt32(tileC)]
        let smemBytes = tileC * 2
        let pipe = KernelCache.shared.pipeline("conv3d_1x1x1")
        MetalContext.shared.run { enc in
            enc.setComputePipelineState(pipe)
            enc.setBuffer(x.buffer, offset: 0, index: 0)
            enc.setBuffer(weight.buffer, offset: 0, index: 1)
            enc.setBuffer(bias.buffer, offset: 0, index: 2)
            enc.setBuffer(out.buffer, offset: 0, index: 3)
            enc.setBytes(&params, length: params.count * 4, index: 4)
            enc.setThreadgroupMemoryLength(smemBytes, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: spatialSize, height: B, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }
        return out
    }

    let kSize = kD * kH * kW
    let spatialOut = D_out * H_out * W_out
    let M = B * D_out * H_out * W_out

    // Gather-GEMM: for each of kSize kernel positions, gather input slice,
    // extract weight slice, GEMM, and accumulate. Chunked to limit memory.
    let out = Tensor.empty([B, C_out, D_out, H_out, W_out], dtype: .float16)
    let ctx = MetalContext.shared

    let chunkSize = min(M, 32768) // 32K rows max per chunk
    let gathered = Tensor.empty([chunkSize, C_in], dtype: .float16)
    let wSlice = Tensor.empty([C_out, C_in], dtype: .float16)
    let gemmOutBuf = Tensor.empty([chunkSize, C_out], dtype: .float16)

    let gatherPipe = KernelCache.shared.pipeline("gather_input_3d")
    let wPipe = KernelCache.shared.pipeline("extract_weight_slice")
    let accPipe = KernelCache.shared.pipeline("accumulate_conv_output")

    for fd in 0..<kD {
        for fh in 0..<kH {
            for fw in 0..<kW {
                let kPos = fd * kH * kW + fh * kW + fw

                // Extract weight slice once per kernel position
                var wParams: [UInt32] = [UInt32(C_in), UInt32(kSize), UInt32(kPos)]
                ctx.run { enc in
                    enc.setComputePipelineState(wPipe)
                    enc.setBuffer(weight.buffer, offset: 0, index: 0)
                    enc.setBuffer(wSlice.buffer, offset: 0, index: 1)
                    enc.setBytes(&wParams, length: wParams.count * 4, index: 2)
                    enc.dispatchThreads(MTLSize(width: C_in, height: C_out, depth: 1),
                                       threadsPerThreadgroup: MTLSize(width: min(256, C_in), height: 1, depth: 1))
                }

                // Process M rows in chunks
                var rowOff = 0
                while rowOff < M {
                    let rows = min(chunkSize, M - rowOff)

                    var gatherParams: [UInt32] = [
                        UInt32(C_in), UInt32(D), UInt32(H), UInt32(W),
                        UInt32(D_out), UInt32(H_out), UInt32(W_out),
                        UInt32(fd), UInt32(fh), UInt32(fw),
                        UInt32(sD), UInt32(sH), UInt32(sW),
                        UInt32(pD), UInt32(pH), UInt32(pW),
                        UInt32(rowOff)
                    ]
                    ctx.run { enc in
                        enc.setComputePipelineState(gatherPipe)
                        enc.setBuffer(x.buffer, offset: 0, index: 0)
                        enc.setBuffer(gathered.buffer, offset: 0, index: 1)
                        enc.setBytes(&gatherParams, length: gatherParams.count * 4, index: 2)
                        enc.dispatchThreads(MTLSize(width: C_in, height: rows, depth: 1),
                                           threadsPerThreadgroup: MTLSize(width: min(256, C_in), height: 1, depth: 1))
                    }

                    _ = simpleMatmulF16(gathered, wSlice, M: rows, K: C_in, N: C_out, into: gemmOutBuf)

                    let isFirst = (kPos == 0)
                    var accParams: [UInt32] = [UInt32(C_out), UInt32(spatialOut), isFirst ? 1 : 0, UInt32(rowOff)]
                    ctx.run { enc in
                        enc.setComputePipelineState(accPipe)
                        enc.setBuffer(gemmOutBuf.buffer, offset: 0, index: 0)
                        enc.setBuffer(bias.buffer, offset: 0, index: 1)
                        enc.setBuffer(out.buffer, offset: 0, index: 2)
                        enc.setBytes(&accParams, length: accParams.count * 4, index: 3)
                        enc.dispatchThreads(MTLSize(width: rows * C_out, height: 1, depth: 1),
                                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                    }

                    rowOff += rows
                }
            }
        }
    }
    return out
}


/// Pixel norm: x / sqrt(mean(x^2, dim=C) + eps). Input/Output: [B, C, D, H, W].
public func pixelNorm3d(_ x: Tensor, B: Int, C: Int, spatialSize: Int, eps: Float = 1e-8) -> Tensor {
    let out = Tensor.empty(x.shape, dtype: .float16)
    var c = UInt32(C)
    var s = UInt32(spatialSize)
    var e = eps
    let pipe = KernelCache.shared.pipeline("pixel_norm_3d")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&c, length: 4, index: 2)
        enc.setBytes(&s, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: spatialSize, height: B, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(256, spatialSize), height: 1, depth: 1))
    }
    return out
}

/// Group norm 3D with learned weight+bias. Input: [B, C, D, H, W].
public func groupNorm3d(_ x: Tensor, weight: Tensor, bias: Tensor,
                        B: Int, C: Int, spatialSize: Int,
                        numGroups: Int, eps: Float = 1e-6) -> Tensor {
    let out = Tensor.empty(x.shape, dtype: .float16)
    var c = UInt32(C)
    var s = UInt32(spatialSize)
    var ng = UInt32(numGroups)
    var e = eps
    let pipe = KernelCache.shared.pipeline("group_norm_3d")
    let tgSize = min(256, C / numGroups * spatialSize)
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: 0, index: 1)
        enc.setBuffer(bias.buffer, offset: 0, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        enc.setBytes(&c, length: 4, index: 4)
        enc.setBytes(&s, length: 4, index: 5)
        enc.setBytes(&ng, length: 4, index: 6)
        enc.setBytes(&e, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: B, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    return out
}

/// Pixel shuffle 3D: [B, C*p1*p2*p3, D, H, W] → [B, C, D*p1, H*p2, W*p3]
public func pixelShuffle3d(_ x: Tensor, B: Int, C_out: Int, D_in: Int, H_in: Int, W_in: Int,
                           p1: Int, p2: Int, p3: Int) -> Tensor {
    let D_out = D_in * p1, H_out = H_in * p2, W_out = W_in * p3
    let out = Tensor.empty([B, C_out, D_out, H_out, W_out], dtype: .float16)
    var params: [UInt32] = [UInt32(C_out), UInt32(D_out), UInt32(H_out), UInt32(W_out),
                            UInt32(p1), UInt32(p2), UInt32(p3), UInt32(B)]
    let total = B * C_out * D_out * H_out * W_out
    let pipe = KernelCache.shared.pipeline("pixel_shuffle_3d")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&params, length: params.count * 4, index: 2)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Scale-shift (AdaLN): out = x * (1 + scale) + shift
/// x: [B, C, D, H, W], scale/shift: [B, C] broadcast over spatial dims.
public func scaleShift3d(_ x: Tensor, scale: Tensor, shift: Tensor,
                         C: Int, spatialSize: Int) -> Tensor {
    let out = Tensor.empty(x.shape, dtype: .float16)
    var c = UInt32(C)
    var s = UInt32(spatialSize)
    let pipe = KernelCache.shared.pipeline("scale_shift_3d")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(scale.buffer, offset: 0, index: 1)
        enc.setBuffer(shift.buffer, offset: 0, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        enc.setBytes(&c, length: 4, index: 4)
        enc.setBytes(&s, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: x.count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Causal pad: repeat first temporal frame `pad` times at the front.
/// Input: [B, C, D, H, W] → Output: [B, C, D+pad, H, W].
public func causalPad(_ x: Tensor, B: Int, C: Int, D: Int, H: Int, W: Int, pad: Int) -> Tensor {
    let D_out = D + pad
    let out = Tensor.empty([B, C, D_out, H, W], dtype: .float16)
    var c = UInt32(C), d = UInt32(D), h = UInt32(H), w = UInt32(W), p = UInt32(pad)
    let total = B * C * D_out * H * W
    let pipe = KernelCache.shared.pipeline("causal_pad_repeat")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&c, length: 4, index: 2)
        enc.setBytes(&d, length: 4, index: 3)
        enc.setBytes(&h, length: 4, index: 4)
        enc.setBytes(&w, length: 4, index: 5)
        enc.setBytes(&p, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Pad by repeating last frame at the end. Input [B, C, D, H, W] → [B, C, D+pad, H, W].
public func causalPadBack(_ x: Tensor, B: Int, C: Int, D: Int, H: Int, W: Int, pad: Int) -> Tensor {
    let D_out = D + pad
    let out = Tensor.empty([B, C, D_out, H, W], dtype: .float16)
    var c = UInt32(C), d = UInt32(D), h = UInt32(H), w = UInt32(W), p = UInt32(pad)
    let total = B * C * D_out * H * W
    let pipe = KernelCache.shared.pipeline("causal_pad_back")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&c, length: 4, index: 2)
        enc.setBytes(&d, length: 4, index: 3)
        enc.setBytes(&h, length: 4, index: 4)
        enc.setBytes(&w, length: 4, index: 5)
        enc.setBytes(&p, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Channel scale-shift: out = x * scale[c] + shift[c], broadcast over batch and spatial dims.
/// x: [B, C, ...spatial], scale/shift: [C]
public func channelScaleShift(_ x: Tensor, scale: Tensor, shift: Tensor, C: Int, spatialSize: Int) -> Tensor {
    let out = Tensor.empty(x.shape, dtype: .float16)
    var cU = UInt32(C), sp = UInt32(spatialSize)
    let total = x.count
    let pipe = KernelCache.shared.pipeline("channel_scale_shift")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(scale.buffer, offset: 0, index: 1)
        enc.setBuffer(shift.buffer, offset: 0, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        enc.setBytes(&cU, length: 4, index: 4)
        enc.setBytes(&sp, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// AdaLN combine (4 outputs) on GPU: table[4,C] + tsEmb[B,4C] → 4 separate [B,C] tensors.
/// Returns (out0, out1, out2, out3) = (shift1, scale1, shift2, scale2).
public func adaLNCombine4(_ tsEmb: Tensor, table: Tensor, B: Int, C: Int) -> (Tensor, Tensor, Tensor, Tensor) {
    let total = B * C
    let out0 = Tensor.empty([B, C], dtype: .float16)
    let out1 = Tensor.empty([B, C], dtype: .float16)
    let out2 = Tensor.empty([B, C], dtype: .float16)
    let out3 = Tensor.empty([B, C], dtype: .float16)
    var c = UInt32(C)
    let pipe = KernelCache.shared.pipeline("ada_ln_combine4")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(table.buffer, offset: 0, index: 0)
        enc.setBuffer(tsEmb.buffer, offset: 0, index: 1)
        enc.setBuffer(out0.buffer, offset: 0, index: 2)
        enc.setBuffer(out1.buffer, offset: 0, index: 3)
        enc.setBuffer(out2.buffer, offset: 0, index: 4)
        enc.setBuffer(out3.buffer, offset: 0, index: 5)
        enc.setBytes(&c, length: 4, index: 6)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return (out0, out1, out2, out3)
}

/// AdaLN combine (2 outputs) on GPU: table[2,C] + tsEmb[B,2C] → 2 separate [B,C] tensors.
/// Returns (shift, scale).
public func adaLNCombine2(_ tsEmb: Tensor, table: Tensor, B: Int, C: Int) -> (Tensor, Tensor) {
    let total = B * C
    let out0 = Tensor.empty([B, C], dtype: .float16)
    let out1 = Tensor.empty([B, C], dtype: .float16)
    var c = UInt32(C)
    let pipe = KernelCache.shared.pipeline("ada_ln_combine2")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(table.buffer, offset: 0, index: 0)
        enc.setBuffer(tsEmb.buffer, offset: 0, index: 1)
        enc.setBuffer(out0.buffer, offset: 0, index: 2)
        enc.setBuffer(out1.buffer, offset: 0, index: 3)
        enc.setBytes(&c, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return (out0, out1)
}

/// Repeat channels on GPU: [B, C, spatial] → [B, C*repeats, spatial]
public func repeatChannelsGPU(_ x: Tensor, B: Int, C: Int, spatialSize: Int, repeats: Int) -> Tensor {
    let outC = C * repeats
    let total = B * outC * spatialSize
    let out = Tensor.empty([B, outC, spatialSize], dtype: .float16)
    var cU = UInt32(C), sp = UInt32(spatialSize), rp = UInt32(repeats)
    let pipe = KernelCache.shared.pipeline("repeat_channels")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&cU, length: 4, index: 2)
        enc.setBytes(&sp, length: 4, index: 3)
        enc.setBytes(&rp, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return Tensor(buffer: out.buffer, shape: [B, outC, spatialSize], dtype: .float16)
}

/// Temporal slice on GPU: x[:, :, from:, :, :] → [B, C, D-from, H, W]
public func temporalSliceGPU(_ x: Tensor, B: Int, C: Int, D: Int, H: Int, W: Int, from: Int) -> Tensor {
    let D_out = D - from
    let HW = H * W
    let total = B * C * D_out * HW
    let out = Tensor.empty([B, C, D_out, H, W], dtype: .float16)
    var d = UInt32(D), dOut = UInt32(D_out), hw = UInt32(HW), f = UInt32(from)
    let pipe = KernelCache.shared.pipeline("temporal_slice")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&d, length: 4, index: 2)
        enc.setBytes(&dOut, length: 4, index: 3)
        enc.setBytes(&hw, length: 4, index: 4)
        enc.setBytes(&f, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// SiLU (Swish) for 3D tensors — reuses existing kernel.
public func silu3d(_ x: Tensor) -> Tensor {
    return silu(x)
}

/// Unpatchify 3D: [B, C*p*p, F, H, W] → [B, C, F, H*p, W*p] (patch_size_t=1)
public func unpatchify3d(_ x: Tensor, B: Int, C_out: Int, F: Int, H_in: Int, W_in: Int, patchSize: Int) -> Tensor {
    let H_out = H_in * patchSize, W_out = W_in * patchSize
    let out = Tensor.empty([B, C_out, F, H_out, W_out], dtype: .float16)
    var params: [UInt32] = [UInt32(C_out), UInt32(F), UInt32(H_out), UInt32(W_out), UInt32(patchSize), UInt32(B)]
    let total = B * C_out * F * H_out * W_out
    let pipe = KernelCache.shared.pipeline("unpatchify_3d")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&params, length: params.count * 4, index: 2)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Timestep embedding (sinusoidal). Input: [B] float, Output: [B, dim] f16.
public func timestepEmbedding(_ timesteps: Tensor, B: Int, dim: Int) -> Tensor {
    let out = Tensor.empty([B, dim], dtype: .float16)
    var d = UInt32(dim)
    let pipe = KernelCache.shared.pipeline("timestep_embedding")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(timesteps.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&d, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: dim, height: B, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(256, dim), height: 1, depth: 1))
    }
    return out
}

/// Linear layer (f16): out = x @ W^T + bias. x: [B, K], W: [N, K], out: [B, N].
public func linearF16(_ x: Tensor, weight: Tensor, bias: Tensor, B: Int, K: Int, N: Int) -> Tensor {
    let out = Tensor.empty([B, N], dtype: .float16)
    var k = UInt32(K), n = UInt32(N)
    let pipe = KernelCache.shared.pipeline("linear_f16")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: 0, index: 1)
        enc.setBuffer(bias.buffer, offset: 0, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        enc.setBytes(&k, length: 4, index: 4)
        enc.setBytes(&n, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: N, height: B, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(256, N), height: 1, depth: 1))
    }
    return out
}

private func sampleFromLogits(_ logits: inout [Float], temperature: Float, topP: Float) -> Int {
    let n = logits.count

    // Apply temperature
    for i in 0..<n { logits[i] /= temperature }

    // Softmax
    var maxVal: Float = -.greatestFiniteMagnitude
    for i in 0..<n { maxVal = max(maxVal, logits[i]) }
    var sumExp: Float = 0
    for i in 0..<n {
        logits[i] = expf(logits[i] - maxVal)
        sumExp += logits[i]
    }
    for i in 0..<n { logits[i] /= sumExp }

    // Sort by probability descending
    var indices = Array(0..<n)
    indices.sort { logits[$0] > logits[$1] }

    // Top-p: keep smallest set with cumulative prob >= topP
    var cumProb: Float = 0
    var cutoff = n
    for i in 0..<n {
        cumProb += logits[indices[i]]
        if cumProb >= topP {
            cutoff = i + 1
            break
        }
    }

    // Renormalize and sample
    var r = Float.random(in: 0..<1) * cumProb
    for i in 0..<cutoff {
        r -= logits[indices[i]]
        if r <= 0 { return indices[i] }
    }
    return indices[0]
}
