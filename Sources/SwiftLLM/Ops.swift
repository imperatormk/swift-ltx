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

/// f16×f16 → f32 GEMM via FlashAttention. A: [M,K] f16, B^T: [N,K] f16, C: [M,N] f32.
public func matmulF16toF32(
    _ a: Tensor, _ b: Tensor,
    M: Int, K: Int, N: Int
) -> MTLBuffer {
    var gemmDesc = GEMMDescriptor()
    gemmDesc.matrixDimensions = (M: UInt32(M), N: UInt32(N), K: UInt32(K))
    gemmDesc.memoryPrecisions = (A: .FP16, B: .FP16, C: .FP32)
    gemmDesc.transposeState = (A: false, B: true)
    gemmDesc.quantizedB = false

    let (kernel, pipeline) = GEMMKernel.pipeline(for: gemmDesc)
    let bufLen = M * N * 4
    let out = MetalContext.shared.device.makeBuffer(length: bufLen, options: .storageModeShared)!

    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipeline)
        enc.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
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

/// f32×f32 → f32 GEMM via FlashAttention. A: [M,K] f32, B^T: [N,K] f32, C: [M,N] f32.
public func matmulF32xF32(
    _ a: MTLBuffer, _ b: MTLBuffer,
    M: Int, K: Int, N: Int
) -> MTLBuffer {
    var gemmDesc = GEMMDescriptor()
    gemmDesc.matrixDimensions = (M: UInt32(M), N: UInt32(N), K: UInt32(K))
    gemmDesc.memoryPrecisions = (A: .FP32, B: .FP32, C: .FP32)
    gemmDesc.transposeState = (A: false, B: true)
    gemmDesc.quantizedB = false

    let (kernel, pipeline) = GEMMKernel.pipeline(for: gemmDesc)
    let out = MetalContext.shared.device.makeBuffer(length: M * N * 4, options: .storageModeShared)!

    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipeline)
        enc.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
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

/// f32×f16 → f32 GEMM via FlashAttention. A: [M,K] f32, B^T: [N,K] f16, C: [M,N] f32.
public func matmulF32xF16(
    _ a: MTLBuffer, _ b: Tensor,
    M: Int, K: Int, N: Int
) -> MTLBuffer {
    var gemmDesc = GEMMDescriptor()
    gemmDesc.matrixDimensions = (M: UInt32(M), N: UInt32(N), K: UInt32(K))
    gemmDesc.memoryPrecisions = (A: .FP32, B: .FP16, C: .FP32)
    gemmDesc.transposeState = (A: false, B: true)
    gemmDesc.quantizedB = false

    let (kernel, pipeline) = GEMMKernel.pipeline(for: gemmDesc)
    let out = MetalContext.shared.device.makeBuffer(length: M * N * 4, options: .storageModeShared)!

    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipeline)
        enc.setThreadgroupMemoryLength(Int(kernel.threadgroupMemoryAllocation), index: 0)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
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

/// GEMM with f32 output: C[M,N] = A[M,K] × B^T[N,K]. A/B are f16, C is f32.
public func simpleMatmulF16toF32(
    _ a: Tensor, _ b: Tensor,
    M: Int, K: Int, N: Int
) -> MTLBuffer {
    let ctx = MetalContext.shared
    let out = ctx.device.makeBuffer(length: M * N * 4, options: .storageModeShared)!
    let pipe = KernelCache.shared.pipeline("simple_gemm_f16_f32out")
    var m = UInt32(M), n = UInt32(N), k = UInt32(K)
    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
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

/// Add f16 bias to f32 buffer in-place: buf[row*N + col] += bias[col]
public func addBiasF32(_ buf: MTLBuffer, bias: Tensor, M: Int, N: Int) {
    let pipe = KernelCache.shared.pipeline("broadcast_bias_add_f32")
    var n = UInt32(N)
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBuffer(bias.buffer, offset: 0, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: M * N, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}

/// Cast f32 buffer → f16 Tensor
public func castF32toF16(_ buf: MTLBuffer, count: Int, shape: [Int]) -> Tensor {
    let out = Tensor.empty(shape, dtype: .float16)
    let pipe = KernelCache.shared.pipeline("cast_f32_to_f16")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Cast f16 Tensor → f32 MTLBuffer
public func castF16toF32(_ x: Tensor) -> MTLBuffer {
    let count = x.count
    let buf = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
    let pipe = KernelCache.shared.pipeline("cast_f16_to_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(buf, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return buf
}

/// In-place add f16 tensor to f32 buffer: acc[i] += x[i]
public func elemAddF32F16(_ acc: MTLBuffer, _ x: Tensor) {
    let pipe = KernelCache.shared.pipeline("elem_add_f32_f16")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(acc, offset: 0, index: 0)
        enc.setBuffer(x.buffer, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: x.count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}

/// In-place add f32: acc[i] += b[i]
public func elemAddF32(_ acc: MTLBuffer, _ b: MTLBuffer, count: Int) {
    let pipe = KernelCache.shared.pipeline("elem_add_f32_inplace")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(acc, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}

/// RMS Norm with f32 input → f16 output. Avoids clipping large f32 residual values.
public func rmsNormF32in(_ x: MTLBuffer, weight: Tensor, eps: Float, dim: Int, rows: Int) -> Tensor {
    let out = Tensor.empty([rows, dim], dtype: .float16)
    let pipe = KernelCache.shared.pipeline("rms_norm_f32in")
    var d = UInt32(dim)
    var e = eps
    let tgSize = min(256, dim)
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        enc.setBytes(&d, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    return out
}

/// AdaLN modulate with f32 input: out = x * (1+scale) + shift. Output f16.
public func adalnModulateF32in(_ x: MTLBuffer, scale: Tensor, shift: Tensor, count: Int) -> Tensor {
    let out = Tensor.empty(scale.shape, dtype: .float16)
    let pipe = KernelCache.shared.pipeline("adaln_modulate_f32in")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(scale.buffer, offset: 0, index: 1)
        enc.setBuffer(shift.buffer, offset: 0, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Element-wise multiply f32 * f16 → f16 output
public func elemMulF32inF16(_ a: MTLBuffer, _ b: Tensor, count: Int) -> Tensor {
    let out = Tensor.empty(b.shape, dtype: .float16)
    let pipe = KernelCache.shared.pipeline("elem_mul_f32in_f16")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Element-wise multiply: out[i] = a_f32[i] * b_f16[i]
public func elemMulF32F16(_ a: MTLBuffer, _ b: Tensor, count: Int) -> MTLBuffer {
    let out = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
    let pipe = KernelCache.shared.pipeline("elem_mul_f32_f16")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
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

// MARK: - DiT Ops

/// GELU approximate (tanh variant).
public func geluApproximate(_ x: Tensor) -> Tensor {
    let out = Tensor.empty(x.shape, dtype: .float16)
    let pipe = KernelCache.shared.pipeline("gelu_approximate")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: x.count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Fused GEGLU: gelu_approx(a) * b. a and b must have same shape.
public func geglu(_ a: Tensor, _ b: Tensor) -> Tensor {
    let out = Tensor.empty(a.shape, dtype: .float16)
    let pipe = KernelCache.shared.pipeline("geglu")
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

/// Layer Norm: (x - mean) / sqrt(var + eps) * weight + bias.
/// x: [..., dim], weight/bias: [dim] f16.
/// Layer norm with f32 input → f16 output.
public func layerNormF32in(_ x: MTLBuffer, weight: Tensor, bias: Tensor, dim: Int, rows: Int, eps: Float = 1e-6) -> Tensor {
    let out = Tensor.empty([rows, dim], dtype: .float16)
    var d = UInt32(dim)
    var e = eps
    let tgSize = min(256, dim)
    let pipe = KernelCache.shared.pipeline("layer_norm_f32in")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: 0, index: 1)
        enc.setBuffer(bias.buffer, offset: 0, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        enc.setBytes(&d, length: 4, index: 4)
        enc.setBytes(&e, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    return out
}

public func layerNorm(_ x: Tensor, weight: Tensor, bias: Tensor, dim: Int, eps: Float = 1e-6) -> Tensor {
    let rows = x.count / dim
    let out = Tensor.empty(x.shape, dtype: .float16)
    var d = UInt32(dim)
    var e = eps
    let tgSize = min(256, dim)
    let pipe = KernelCache.shared.pipeline("layer_norm")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: 0, index: 1)
        enc.setBuffer(bias.buffer, offset: 0, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        enc.setBytes(&d, length: 4, index: 4)
        enc.setBytes(&e, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    return out
}

/// AdaLN modulate: out = x * (1 + scale) + shift. All same shape.
public func adalnModulate(_ x: Tensor, scale: Tensor, shift: Tensor) -> Tensor {
    let out = Tensor.empty(x.shape, dtype: .float16)
    let pipe = KernelCache.shared.pipeline("adaln_modulate")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(scale.buffer, offset: 0, index: 1)
        enc.setBuffer(shift.buffer, offset: 0, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: x.count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

// MARK: - DiT F32 Ops

/// SiLU f32: out = x / (1 + exp(-x))
public func siluF32(_ x: MTLBuffer, count: Int) -> MTLBuffer {
    let out = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
    let pipe = KernelCache.shared.pipeline("silu_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Element-wise multiply f32
public func elemMulF32(_ a: MTLBuffer, _ b: MTLBuffer, count: Int) -> MTLBuffer {
    let out = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
    let pipe = KernelCache.shared.pipeline("mul_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// 1 + x element-wise f32
public func onePlusF32(_ x: MTLBuffer, count: Int) -> MTLBuffer {
    let out = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
    let pipe = KernelCache.shared.pipeline("one_plus_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Broadcast [B, dim] → [B, numTokens, dim] f32
public func broadcastToSeqF32(_ x: MTLBuffer, numTokens: Int, dim: Int, B: Int) -> MTLBuffer {
    if numTokens == 1 { return x }
    let total = B * numTokens * dim
    let out = MetalContext.shared.device.makeBuffer(length: total * 4, options: .storageModeShared)!
    var n = UInt32(numTokens), d = UInt32(dim)
    let pipe = KernelCache.shared.pipeline("broadcast_seq_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.setBytes(&d, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// AdaLN modulate f32: out = x * (1 + scale) + shift
public func adalnModulateF32(_ x: MTLBuffer, scale: MTLBuffer, shift: MTLBuffer, count: Int) -> MTLBuffer {
    let out = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
    let pipe = KernelCache.shared.pipeline("adaln_modulate_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(scale, offset: 0, index: 1)
        enc.setBuffer(shift, offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// 3D RoPE f32
public func rope3DF32(_ x: MTLBuffer, cosFreqs: MTLBuffer, sinFreqs: MTLBuffer, dim: Int, count: Int) -> MTLBuffer {
    let out = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
    let batchSeq = count / dim
    let pairs = dim / 2
    var d = UInt32(dim)
    let pipe = KernelCache.shared.pipeline("rope_3d_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBuffer(cosFreqs, offset: 0, index: 2)
        enc.setBuffer(sinFreqs, offset: 0, index: 3)
        enc.setBytes(&d, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: pairs, height: batchSeq, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(pairs, 256), height: 1, depth: 1))
    }
    return out
}

/// Transpose [seqLen, nHeads, headDim] → [nHeads, seqLen, headDim] f32
public func transposeSHF32(_ x: MTLBuffer, seqLen: Int, nHeads: Int, headDim: Int) -> MTLBuffer {
    let count = seqLen * nHeads * headDim
    let out = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
    var sl = UInt32(seqLen), nh = UInt32(nHeads), hd = UInt32(headDim)
    let pipe = KernelCache.shared.pipeline("transpose_sh_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&sl, length: 4, index: 2)
        enc.setBytes(&nh, length: 4, index: 3)
        enc.setBytes(&hd, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: headDim, height: seqLen * nHeads, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1))
    }
    return out
}

/// Transpose [nHeads, seqLen, headDim] → [seqLen, nHeads, headDim] f32
public func transposeHSF32(_ x: MTLBuffer, seqLen: Int, nHeads: Int, headDim: Int) -> MTLBuffer {
    let count = nHeads * seqLen * headDim
    let out = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
    var sl = UInt32(seqLen), nh = UInt32(nHeads), hd = UInt32(headDim)
    let pipe = KernelCache.shared.pipeline("transpose_hs_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&sl, length: 4, index: 2)
        enc.setBytes(&nh, length: 4, index: 3)
        enc.setBytes(&hd, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: headDim, height: nHeads * seqLen, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1))
    }
    return out
}

/// RMS Norm f32 input → f32 output
public func rmsNormF32(_ x: MTLBuffer, weight: Tensor, eps: Float, dim: Int, rows: Int) -> MTLBuffer {
    let out = MetalContext.shared.device.makeBuffer(length: rows * dim * 4, options: .storageModeShared)!
    let pipe = KernelCache.shared.pipeline("rms_norm_f32")
    var d = UInt32(dim)
    var e = eps
    let tgSize = min(256, dim)
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setBytes(&d, length: 4, index: 3)
        enc.setBytes(&e, length: 4, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    return out
}

/// Layer norm f32 input → f32 output. Weight/bias must be f32.
public func layerNormF32(_ x: MTLBuffer, weight: MTLBuffer, bias: MTLBuffer, dim: Int, rows: Int, eps: Float = 1e-6) -> MTLBuffer {
    let out = MetalContext.shared.device.makeBuffer(length: rows * dim * 4, options: .storageModeShared)!
    var d = UInt32(dim)
    var e = eps
    let tgSize = min(256, dim)
    let pipe = KernelCache.shared.pipeline("layer_norm_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(weight, offset: 0, index: 1)
        enc.setBuffer(bias, offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        enc.setBytes(&d, length: 4, index: 4)
        enc.setBytes(&e, length: 4, index: 5)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    return out
}

/// Timestep embedding f32 output. Input: [B] float, Output: [B, dim] f32 MTLBuffer.
public func timestepEmbeddingF32(_ timesteps: Tensor, B: Int, dim: Int) -> MTLBuffer {
    let out = MetalContext.shared.device.makeBuffer(length: B * dim * 4, options: .storageModeShared)!
    var d = UInt32(dim)
    let pipe = KernelCache.shared.pipeline("timestep_embedding_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(timesteps.buffer, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&d, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: dim, height: B, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(256, dim), height: 1, depth: 1))
    }
    return out
}

/// GELU approximate f32 (wrapper for MTLBuffer API)
public func geluApproximateF32(_ x: MTLBuffer, count: Int) -> MTLBuffer {
    let out = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
    let pipe = KernelCache.shared.pipeline("gelu_approximate_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// GEGLU f32 (wrapper for MTLBuffer API)
public func gegluF32(_ gate: MTLBuffer, _ value: MTLBuffer, count: Int) -> MTLBuffer {
    let out = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
    let pipe = KernelCache.shared.pipeline("geglu_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(gate, offset: 0, index: 0)
        enc.setBuffer(value, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// AdaLN combine2 f32: table[2,C] (f16) + tsEmb[B,2C] (f32) → 2 separate [B,C] f32 buffers.
/// NOTE: tsEmb must be [B, 2*C] layout where each batch has 2*C elements.
public func adaLNCombine2F32(_ tsEmb: MTLBuffer, table: Tensor, B: Int, C: Int) -> (MTLBuffer, MTLBuffer) {
    let total = B * C
    let out0 = MetalContext.shared.device.makeBuffer(length: total * 4, options: .storageModeShared)!
    let out1 = MetalContext.shared.device.makeBuffer(length: total * 4, options: .storageModeShared)!
    var c = UInt32(C)
    let pipe = KernelCache.shared.pipeline("ada_ln_combine2_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(table.buffer, offset: 0, index: 0)
        enc.setBuffer(tsEmb, offset: 0, index: 1)
        enc.setBuffer(out0, offset: 0, index: 2)
        enc.setBuffer(out1, offset: 0, index: 3)
        enc.setBytes(&c, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return (out0, out1)
}

/// AdaLN combine2 shared f32: table[2,C] (f16) + tsEmb[B,C] (f32, SHARED for both outputs) → 2 × [B,C] f32.
/// Use when tsEmb is [B,C] (not [B,2C]) — both outputs add the same tsEmb, differ only by table row.
public func adaLNCombine2SharedF32(_ tsEmb: MTLBuffer, table: Tensor, B: Int, C: Int) -> (MTLBuffer, MTLBuffer) {
    let total = B * C
    let out0 = MetalContext.shared.device.makeBuffer(length: total * 4, options: .storageModeShared)!
    let out1 = MetalContext.shared.device.makeBuffer(length: total * 4, options: .storageModeShared)!
    var c = UInt32(C)
    let pipe = KernelCache.shared.pipeline("ada_ln_combine2_shared_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(table.buffer, offset: 0, index: 0)
        enc.setBuffer(tsEmb, offset: 0, index: 1)
        enc.setBuffer(out0, offset: 0, index: 2)
        enc.setBuffer(out1, offset: 0, index: 3)
        enc.setBytes(&c, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return (out0, out1)
}

/// Add f32 bias (f32) to f32 buffer in-place
public func addBiasF32F32(_ buf: MTLBuffer, bias: MTLBuffer, M: Int, N: Int) {
    let pipe = KernelCache.shared.pipeline("broadcast_bias_add_f32_f32")
    var n = UInt32(N)
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(buf, offset: 0, index: 0)
        enc.setBuffer(bias, offset: 0, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: M * N, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
}

// MARK: - F32 Pipeline Ops (Phase 3)

/// Per-channel scale+bias f32: out = x * scale[c] + bias[c]. All f32 MTLBuffers.
public func channelScaleBiasF32(x: MTLBuffer, scale: MTLBuffer, bias: MTLBuffer,
                                 B: Int, C: Int, spatial: Int) -> MTLBuffer {
    let total = B * C * spatial
    let out = MetalContext.shared.device.makeBuffer(length: total * 4, options: .storageModeShared)!
    let pipe = KernelCache.shared.pipeline("channel_scale_bias_f32")
    var cU = UInt32(C), sU = UInt32(spatial)
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(scale, offset: 0, index: 1)
        enc.setBuffer(bias, offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        enc.setBytes(&cU, length: 4, index: 4)
        enc.setBytes(&sU, length: 4, index: 5)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// 3D Convolution f32 — all buffers are f32.
public func conv3dF32(_ x: MTLBuffer, weight: MTLBuffer, bias: MTLBuffer,
                       B: Int, C_in: Int, D: Int, H: Int, W: Int,
                       C_out: Int, kD: Int, kH: Int, kW: Int,
                       sD: Int = 1, sH: Int = 1, sW: Int = 1,
                       pD: Int = 0, pH: Int = 0, pW: Int = 0) -> MTLBuffer {
    let D_out = (D + 2*pD - kD) / sD + 1
    let H_out = (H + 2*pH - kH) / sH + 1
    let W_out = (W + 2*pW - kW) / sW + 1
    let outCount = B * C_out * D_out * H_out * W_out

    let isPointwise = kD == 1 && kH == 1 && kW == 1 && sD == 1 && sH == 1 && sW == 1 && pD == 0 && pH == 0 && pW == 0

    if isPointwise {
        let out = MetalContext.shared.device.makeBuffer(length: outCount * 4, options: .storageModeShared)!
        let spatialSize = D * H * W
        let tgSize = min(C_out, 256)
        let tileC = min(C_in, 1024)
        var params: [UInt32] = [UInt32(C_in), UInt32(spatialSize), UInt32(C_out), UInt32(tileC)]
        let smemBytes = tileC * 4  // float
        let pipe = KernelCache.shared.pipeline("conv3d_1x1x1_f32")
        MetalContext.shared.run { enc in
            enc.setComputePipelineState(pipe)
            enc.setBuffer(x, offset: 0, index: 0)
            enc.setBuffer(weight, offset: 0, index: 1)
            enc.setBuffer(bias, offset: 0, index: 2)
            enc.setBuffer(out, offset: 0, index: 3)
            enc.setBytes(&params, length: params.count * 4, index: 4)
            enc.setThreadgroupMemoryLength(smemBytes, index: 0)
            enc.dispatchThreadgroups(MTLSize(width: spatialSize, height: B, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }
        return out
    }

    // General path: im2col + single GEMM + scatter
    let kSize = kD * kH * kW
    let spatialOut = D_out * H_out * W_out
    let M = B * D_out * H_out * W_out
    let K = C_in * kSize
    let out = MetalContext.shared.device.makeBuffer(length: outCount * 4, options: .storageModeShared)!
    let ctx = MetalContext.shared

    // im2col: build [M, C_in * kSize] matrix
    let im2colBuf = ctx.device.makeBuffer(length: M * K * 4, options: .storageModeShared)!
    var im2colParams: [UInt32] = [
        UInt32(C_in), UInt32(D), UInt32(H), UInt32(W),
        UInt32(D_out), UInt32(H_out), UInt32(W_out),
        UInt32(kD), UInt32(kH), UInt32(kW),
        UInt32(sD), UInt32(sH), UInt32(sW),
        UInt32(pD), UInt32(pH), UInt32(pW)
    ]
    let im2colPipe = KernelCache.shared.pipeline("im2col_3d_f32")
    ctx.run { enc in
        enc.setComputePipelineState(im2colPipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(im2colBuf, offset: 0, index: 1)
        enc.setBytes(&im2colParams, length: im2colParams.count * 4, index: 2)
        enc.dispatchThreads(MTLSize(width: K, height: M, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(256, K), height: 1, depth: 1))
    }

    // Single GEMM: [M, K] × [C_out, K]^T → [M, C_out]
    // Weight is [C_out, C_in, kD, kH, kW] = [C_out, K] row-major — already correct for B^T
    let gemmResult = matmulF32xF32(im2colBuf, weight, M: M, K: K, N: C_out)

    // Scatter [M, C_out] row-major → [B, C_out, spatial] NCHW + bias
    var scatterParams: [UInt32] = [UInt32(C_out), UInt32(spatialOut)]
    let scatterPipe = KernelCache.shared.pipeline("scatter_conv_output_f32")
    ctx.run { enc in
        enc.setComputePipelineState(scatterPipe)
        enc.setBuffer(gemmResult, offset: 0, index: 0)
        enc.setBuffer(bias, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setBytes(&scatterParams, length: scatterParams.count * 4, index: 3)
        enc.dispatchThreads(MTLSize(width: M * C_out, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Group norm 3D f32. Input/output f32 MTLBuffer.
public func groupNorm3dF32(_ x: MTLBuffer, weight: Tensor, bias: Tensor,
                            B: Int, C: Int, spatialSize: Int,
                            numGroups: Int, eps: Float = 1e-6) -> MTLBuffer {
    let total = B * C * spatialSize
    let out = MetalContext.shared.device.makeBuffer(length: total * 4, options: .storageModeShared)!
    var c = UInt32(C)
    var s = UInt32(spatialSize)
    var ng = UInt32(numGroups)
    var e = eps
    let pipe = KernelCache.shared.pipeline("group_norm_3d_f32")
    let tgSize = min(256, C / numGroups * spatialSize)
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: 0, index: 1)
        enc.setBuffer(bias.buffer, offset: 0, index: 2)
        enc.setBuffer(out, offset: 0, index: 3)
        enc.setBytes(&c, length: 4, index: 4)
        enc.setBytes(&s, length: 4, index: 5)
        enc.setBytes(&ng, length: 4, index: 6)
        enc.setBytes(&e, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: numGroups, height: B, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    return out
}

/// Pixel shuffle 3D f32: [B, C*p1*p2*p3, D, H, W] → [B, C, D*p1, H*p2, W*p3]
public func pixelShuffle3dF32(_ x: MTLBuffer, B: Int, C_out: Int, D_in: Int, H_in: Int, W_in: Int,
                               p1: Int, p2: Int, p3: Int) -> MTLBuffer {
    let D_out = D_in * p1, H_out = H_in * p2, W_out = W_in * p3
    let total = B * C_out * D_out * H_out * W_out
    let out = MetalContext.shared.device.makeBuffer(length: total * 4, options: .storageModeShared)!
    var params: [UInt32] = [UInt32(C_out), UInt32(D_out), UInt32(H_out), UInt32(W_out),
                            UInt32(p1), UInt32(p2), UInt32(p3), UInt32(B)]
    let pipe = KernelCache.shared.pipeline("pixel_shuffle_3d_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&params, length: params.count * 4, index: 2)
        enc.dispatchThreads(MTLSize(width: total, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Element-wise add f32 (out-of-place): out = a + b
public func elemAddF32OOP(_ a: MTLBuffer, _ b: MTLBuffer, count: Int) -> MTLBuffer {
    let out = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
    let pipe = KernelCache.shared.pipeline("elem_add_f32_oop")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Broadcast f32 RoPE freqs [numTokens, dim] → [B, numTokens, dim] f32.
public func broadcastRoPEF32(_ freqs: MTLBuffer, B: Int, seqLen: Int, dim: Int) -> MTLBuffer {
    if B == 1 { return freqs }
    let bytes = seqLen * dim * 4
    let out = MetalContext.shared.device.makeBuffer(length: B * bytes, options: .storageModeShared)!
    for b in 0..<B {
        memcpy(out.contents() + b * bytes, freqs.contents(), bytes)
    }
    return out
}

/// Flash attention bidirectional with f32 buffers (no f16 conversion needed).
/// Q/K/V: MTLBuffer f32, [nHeads, seqLen, headDim]. Returns f32 MTLBuffer [nHeads, R, D].
public func flashAttentionBidirectionalF32(
    q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
    R: Int, C: Int, D: Int, nHeads: Int
) -> MTLBuffer {
    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionInputs = false
    attDesc.lowPrecisionIntermediates = false
    attDesc.lowPrecisionOutputs = false
    attDesc.matrixDimensions = (row: UInt32(R), column: UInt32(C), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)
    attDesc.causal = false

    let (kernel, pipeline) = AttentionKernel.pipeline(for: attDesc, type: .forward)

    let pool = MetalContext.shared.bufferPool
    let oBytes = nHeads * R * D * 4
    let lBytes = nHeads * R * 4
    let bufO = pool.get(length: oBytes)
    let bufL = pool.get(length: lBytes)
    let dummy = pool.get(length: 16)
    gpuZero(bufO, bytes: oBytes)
    gpuZero(bufL, bytes: lBytes)
    gpuZero(dummy, bytes: 16)

    let d = UInt32(D)
    var params: [UInt32] = [
        UInt32(nHeads), 1,
        UInt32(R) * d, UInt32(C) * d, UInt32(C) * d,
        UInt32(R) * d, UInt32(R), UInt32(R),
        UInt32(R) * d, UInt32(C) * d, UInt32(C) * d, UInt32(R) * d,
        0, UInt32(R), UInt32(C)
    ]
    let batchParams = MetalContext.shared.device.makeBuffer(
        bytes: &params, length: params.count * 4, options: .storageModeShared)!

    MetalContext.shared.run { enc in
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(v, offset: 0, index: 2)
        enc.setBuffer(bufO, offset: 0, index: 3)
        enc.setBuffer(bufL, offset: 0, index: 4)
        enc.setBuffer(dummy, offset: 0, index: 5)
        enc.setBuffer(dummy, offset: 0, index: 6)
        enc.setBuffer(dummy, offset: 0, index: 7)
        enc.setBuffer(dummy, offset: 0, index: 8)
        enc.setBuffer(dummy, offset: 0, index: 9)
        AttentionKernel.dispatch(
            encoder: enc, kernel: kernel, pipeline: pipeline,
            batchedParams: batchParams, parallelizationDimension: R,
            numHeads: nHeads, batchSize: 1)
    }
    return bufO
}

/// 3D RoPE: apply precomputed cos/sin freqs to x.
/// x: [B, seqLen, dim], cosFreqs/sinFreqs: [B, seqLen, dim] (f16).
public func rope3D(_ x: Tensor, cosFreqs: Tensor, sinFreqs: Tensor, dim: Int) -> Tensor {
    let out = Tensor.empty(x.shape, dtype: .float16)
    let batchSeq = x.count / dim
    let pairs = dim / 2
    var d = UInt32(dim)
    let pipe = KernelCache.shared.pipeline("rope_3d")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBuffer(cosFreqs.buffer, offset: 0, index: 2)
        enc.setBuffer(sinFreqs.buffer, offset: 0, index: 3)
        enc.setBytes(&d, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: pairs, height: batchSeq, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(pairs, 256), height: 1, depth: 1))
    }
    return out
}

/// Bidirectional flash attention (no causal mask) for DiT.
/// Q: [nHeads, R, D], K: [nHeads, C, D], V: [nHeads, C, D] — all f16, contiguous.
/// Returns: [nHeads, R, D] f16.
public func flashAttentionBidirectional(
    q: Tensor, k: Tensor, v: Tensor,
    R: Int, C: Int, D: Int, nHeads: Int
) -> Tensor {
    // Cast f16 inputs to f32 for full-precision attention (avoids clipping large activations)
    let qF32 = castF16toF32(q)
    let kF32 = castF16toF32(k)
    let vF32 = castF16toF32(v)

    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionInputs = false
    attDesc.lowPrecisionIntermediates = false
    attDesc.lowPrecisionOutputs = false
    attDesc.matrixDimensions = (row: UInt32(R), column: UInt32(C), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)
    attDesc.causal = false

    let (kernel, pipeline) = AttentionKernel.pipeline(for: attDesc, type: .forward)

    let pool = MetalContext.shared.bufferPool
    let oBytes = nHeads * R * D * 4  // f32 output
    let lBytes = nHeads * R * 4
    let bufO = pool.get(length: oBytes)
    let bufL = pool.get(length: lBytes)
    let dummy = pool.get(length: 16)
    gpuZero(bufO, bytes: oBytes)
    gpuZero(bufL, bytes: lBytes)
    gpuZero(dummy, bytes: 16)

    let d = UInt32(D)
    var params: [UInt32] = [
        UInt32(nHeads),         // 0: numHeads
        1,                      // 1: kvRepeatFactor (no GQA in DiT)
        UInt32(R) * d,          // 2: Q stride
        UInt32(C) * d,          // 3: K stride
        UInt32(C) * d,          // 4: V stride
        UInt32(R) * d,          // 5: O stride
        UInt32(R),              // 6: L stride
        UInt32(R),              // 7: D stride
        UInt32(R) * d,          // 8: dO stride
        UInt32(C) * d,          // 9: dV stride
        UInt32(C) * d,          // 10: dK stride
        UInt32(R) * d,          // 11: dQ stride
        0,                      // 12: causalOffset
        UInt32(R),              // 13: R
        UInt32(C)               // 14: C
    ]
    let batchParams = MetalContext.shared.device.makeBuffer(
        bytes: &params, length: params.count * 4, options: .storageModeShared)!

    MetalContext.shared.run { enc in
        enc.setBuffer(qF32, offset: 0, index: 0)
        enc.setBuffer(kF32, offset: 0, index: 1)
        enc.setBuffer(vF32, offset: 0, index: 2)
        enc.setBuffer(bufO, offset: 0, index: 3)
        enc.setBuffer(bufL, offset: 0, index: 4)
        enc.setBuffer(dummy, offset: 0, index: 5)
        enc.setBuffer(dummy, offset: 0, index: 6)
        enc.setBuffer(dummy, offset: 0, index: 7)
        enc.setBuffer(dummy, offset: 0, index: 8)
        enc.setBuffer(dummy, offset: 0, index: 9)

        AttentionKernel.dispatch(
            encoder: enc,
            kernel: kernel,
            pipeline: pipeline,
            batchedParams: batchParams,
            parallelizationDimension: R,
            numHeads: nHeads,
            batchSize: 1)
    }

    // Cast f32 output back to f16
    return castF32toF16(bufO, count: nHeads * R * D, shape: [nHeads, R, D])
}

/// Cross-attention via flash attention (bidirectional, Q/K may differ in seq length).
/// Same as flashAttentionBidirectional but named for clarity.
public func flashCrossAttention(
    q: Tensor, k: Tensor, v: Tensor,
    R: Int, C: Int, D: Int, nHeads: Int
) -> Tensor {
    return flashAttentionBidirectional(q: q, k: k, v: v, R: R, C: C, D: D, nHeads: nHeads)
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
