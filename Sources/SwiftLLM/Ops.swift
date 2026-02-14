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

// MARK: - Quantized matmul (4-bit)

/// Quantized matmul: out = x @ W^T where W is 4-bit packed.
/// x: [M, K] f16. Returns [M, N] f16.
public func matmulQ4(_ x: Tensor, weight: Tensor, scales: Tensor, biases: Tensor, M: Int, K: Int, N: Int, groupSize: Int) -> Tensor {
    let out = Tensor.empty([M, N], dtype: .float16)
    var k = UInt32(K)
    var gs = UInt32(groupSize)
    var n = UInt32(N)
    let pipe = KernelCache.shared.pipeline("matmul_q4")

    let tgSize = 256
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: 0, index: 1)
        enc.setBuffer(scales.buffer, offset: 0, index: 2)
        enc.setBuffer(biases.buffer, offset: 0, index: 3)
        enc.setBuffer(out.buffer, offset: 0, index: 4)
        enc.setBytes(&k, length: 4, index: 5)
        enc.setBytes(&gs, length: 4, index: 6)
        enc.setBytes(&n, length: 4, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: M * N, height: 1, depth: 1),
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
public func sample(_ x: Tensor, temperature: Float = 0.7, topP: Float = 0.9, repetitionPenalty: Float = 1.1, previousTokens: [Int] = []) -> Int {
    let n = x.count
    var logits = [Float](repeating: 0, count: n)
    if x.dtype == .float16 {
        let ptr = x.buffer.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<n { logits[i] = Float(ptr[i]) }
    } else {
        let ptr = x.buffer.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<n { logits[i] = ptr[i] }
    }

    // Apply repetition penalty to previously generated tokens
    for tok in previousTokens where tok < n {
        if logits[tok] > 0 {
            logits[tok] /= repetitionPenalty
        } else {
            logits[tok] *= repetitionPenalty
        }
    }

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
