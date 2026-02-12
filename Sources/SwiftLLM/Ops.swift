// High-level ops built on Metal kernels + FlashAttention.

import Metal
import FlashAttention

// MARK: - Element-wise Ops

public func rmsNorm(_ x: Tensor, weight: Tensor, eps: Float, dim: Int) -> Tensor {
    let rows = x.count / dim
    let out = Tensor.zeros([rows, dim])
    let ctx = MetalContext.shared
    let pipe = KernelCache.shared.pipeline("rms_norm")

    var d = UInt32(dim)
    var e = eps
    let dBuf = ctx.makePooledBuffer(&d, length: 4)
    let eBuf = ctx.makePooledBuffer(&e, length: 4)

    let tgSize = min(256, dim)
    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        enc.setBuffer(dBuf, offset: 0, index: 3)
        enc.setBuffer(eBuf, offset: 0, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    return out
}

public func silu(_ x: Tensor) -> Tensor {
    let out = Tensor.zeros(x.shape)
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
    let out = Tensor.zeros(a.shape)
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
    let out = Tensor.zeros(a.shape)
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
    let out = Tensor.zeros(a.shape)
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
    let out = Tensor.zeros([dim])
    var tid = UInt32(tokenId)
    var d = UInt32(dim)
    let ctx = MetalContext.shared
    let tidBuf = ctx.makePooledBuffer(&tid, length: 4)
    let dBuf = ctx.makePooledBuffer(&d, length: 4)
    let pipe = KernelCache.shared.pipeline("embedding")

    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(table.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBuffer(tidBuf, offset: 0, index: 2)
        enc.setBuffer(dBuf, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: dim, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Batch embedding lookup: [numTokens] → [numTokens, dim]
/// Transpose [seqLen, nHeads, headDim] → [nHeads, seqLen, headDim]
public func transposeSH(_ x: Tensor, seqLen: Int, nHeads: Int, headDim: Int) -> Tensor {
    let out = Tensor.zeros([nHeads, seqLen, headDim])
    var sl = UInt32(seqLen), nh = UInt32(nHeads), hd = UInt32(headDim)
    let ctx = MetalContext.shared
    let slBuf = ctx.makePooledBuffer(&sl, length: 4)
    let nhBuf = ctx.makePooledBuffer(&nh, length: 4)
    let hdBuf = ctx.makePooledBuffer(&hd, length: 4)
    let pipe = KernelCache.shared.pipeline("transpose_sh")
    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBuffer(slBuf, offset: 0, index: 2)
        enc.setBuffer(nhBuf, offset: 0, index: 3)
        enc.setBuffer(hdBuf, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: headDim, height: seqLen * nHeads, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1))
    }
    return out
}

/// Transpose [nHeads, seqLen, headDim] → [seqLen, nHeads, headDim]
public func transposeHS(_ x: Tensor, seqLen: Int, nHeads: Int, headDim: Int) -> Tensor {
    let out = Tensor.zeros([seqLen, nHeads, headDim])
    var sl = UInt32(seqLen), nh = UInt32(nHeads), hd = UInt32(headDim)
    let ctx = MetalContext.shared
    let slBuf = ctx.makePooledBuffer(&sl, length: 4)
    let nhBuf = ctx.makePooledBuffer(&nh, length: 4)
    let hdBuf = ctx.makePooledBuffer(&hd, length: 4)
    let pipe = KernelCache.shared.pipeline("transpose_hs")
    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBuffer(slBuf, offset: 0, index: 2)
        enc.setBuffer(nhBuf, offset: 0, index: 3)
        enc.setBuffer(hdBuf, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: headDim, height: nHeads * seqLen, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(headDim, 256), height: 1, depth: 1))
    }
    return out
}

/// Batch embedding lookup: [numTokens] → [numTokens, dim]
public func embeddingBatch(table: Tensor, tokenIds: [Int], dim: Int) -> Tensor {
    let n = tokenIds.count
    let out = Tensor.zeros([n, dim])
    var ids = tokenIds.map { UInt32($0) }
    let ctx = MetalContext.shared
    let idsBuf = ctx.makePooledBuffer(&ids, length: n * 4)
    var d = UInt32(dim)
    let dBuf = ctx.makePooledBuffer(&d, length: 4)
    let pipe = KernelCache.shared.pipeline("embedding_batch")

    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(table.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBuffer(idsBuf, offset: 0, index: 2)
        enc.setBuffer(dBuf, offset: 0, index: 3)
        enc.dispatchThreads(MTLSize(width: dim, height: n, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Batch quantized embedding lookup: [numTokens] → [numTokens, K]
public func embeddingQ4Batch(weight: Tensor, scales: Tensor, biases: Tensor, tokenIds: [Int], K: Int, groupSize: Int) -> Tensor {
    let n = tokenIds.count
    let out = Tensor.zeros([n, K])
    var ids = tokenIds.map { UInt32($0) }
    let ctx = MetalContext.shared
    let idsBuf = ctx.makePooledBuffer(&ids, length: n * 4)
    var k = UInt32(K)
    var gs = UInt32(groupSize)
    let kBuf = ctx.makePooledBuffer(&k, length: 4)
    let gsBuf = ctx.makePooledBuffer(&gs, length: 4)
    let pipe = KernelCache.shared.pipeline("embedding_q4_batch")

    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(weight.buffer, offset: 0, index: 0)
        enc.setBuffer(scales.buffer, offset: 0, index: 1)
        enc.setBuffer(biases.buffer, offset: 0, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        enc.setBuffer(idsBuf, offset: 0, index: 4)
        enc.setBuffer(kBuf, offset: 0, index: 5)
        enc.setBuffer(gsBuf, offset: 0, index: 6)
        enc.dispatchThreads(MTLSize(width: K, height: n, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

public func embeddingQ4(weight: Tensor, scales: Tensor, biases: Tensor, tokenId: Int, K: Int, groupSize: Int) -> Tensor {
    let out = Tensor.zeros([K])
    var tid = UInt32(tokenId)
    var k = UInt32(K)
    var gs = UInt32(groupSize)
    let ctx = MetalContext.shared
    let tidBuf = ctx.makePooledBuffer(&tid, length: 4)
    let kBuf = ctx.makePooledBuffer(&k, length: 4)
    let gsBuf = ctx.makePooledBuffer(&gs, length: 4)
    let pipe = KernelCache.shared.pipeline("embedding_q4")

    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(weight.buffer, offset: 0, index: 0)
        enc.setBuffer(scales.buffer, offset: 0, index: 1)
        enc.setBuffer(biases.buffer, offset: 0, index: 2)
        enc.setBuffer(out.buffer, offset: 0, index: 3)
        enc.setBuffer(tidBuf, offset: 0, index: 4)
        enc.setBuffer(kBuf, offset: 0, index: 5)
        enc.setBuffer(gsBuf, offset: 0, index: 6)
        enc.dispatchThreads(MTLSize(width: K, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

public func rope(_ x: Tensor, headDim: Int, seqLen: Int, startPos: Int, theta: Float) -> Tensor {
    // x is [nHeads, seqLen, headDim] — modify in-place (copy first, uses pool)
    let buf = MetalContext.shared.bufferPool.get(length: x.byteCount)
    memcpy(buf.contents(), x.buffer.contents(), x.byteCount)
    let out = Tensor(buffer: buf, shape: x.shape)

    var hd = UInt32(headDim)
    var p = UInt32(startPos)
    var t = theta
    var sl = UInt32(seqLen)
    let ctx = MetalContext.shared
    let hdBuf = ctx.makePooledBuffer(&hd, length: 4)
    let pBuf = ctx.makePooledBuffer(&p, length: 4)
    let tBuf = ctx.makePooledBuffer(&t, length: 4)
    let slBuf = ctx.makePooledBuffer(&sl, length: 4)
    let pipe = KernelCache.shared.pipeline("rope")

    let nPairs = headDim / 2
    let nHeadSeq = x.count / headDim

    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(out.buffer, offset: 0, index: 0)
        enc.setBuffer(hdBuf, offset: 0, index: 1)
        enc.setBuffer(pBuf, offset: 0, index: 2)
        enc.setBuffer(tBuf, offset: 0, index: 3)
        enc.setBuffer(slBuf, offset: 0, index: 4)
        enc.dispatchThreads(MTLSize(width: nPairs, height: nHeadSeq, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: min(nPairs, 32), height: 1, depth: 1))
    }
    return out
}

// MARK: - Quantized matmul (4-bit)

/// Quantized matmul: out = x @ W^T where W is 4-bit packed.
/// x: [1, K] float32. Returns [1, N] float32.
public func matmulQ4(_ x: Tensor, weight: Tensor, scales: Tensor, biases: Tensor, M: Int, K: Int, N: Int, groupSize: Int) -> Tensor {
    let out = Tensor.zeros([M, N])
    var k = UInt32(K)
    var gs = UInt32(groupSize)
    var n = UInt32(N)
    let ctx = MetalContext.shared
    let kBuf = ctx.makePooledBuffer(&k, length: 4)
    let gsBuf = ctx.makePooledBuffer(&gs, length: 4)
    let nBuf = ctx.makePooledBuffer(&n, length: 4)
    let pipe = KernelCache.shared.pipeline("matmul_q4")

    let tgSize = 256
    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(weight.buffer, offset: 0, index: 1)
        enc.setBuffer(scales.buffer, offset: 0, index: 2)
        enc.setBuffer(biases.buffer, offset: 0, index: 3)
        enc.setBuffer(out.buffer, offset: 0, index: 4)
        enc.setBuffer(kBuf, offset: 0, index: 5)
        enc.setBuffer(gsBuf, offset: 0, index: 6)
        enc.setBuffer(nBuf, offset: 0, index: 7)
        enc.dispatchThreadgroups(MTLSize(width: M * N, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    return out
}

// MARK: - Fast quantized GEMM (monolithic IR)

/// Fast quantized matmul via monolithic IR kernel.
/// A: [M, K] f32 (input activations). W: [N, K/8] uint32 (packed weights).
/// scales/biases: [N, K/groupSize] f16. Output: [M, N] f32.
public func matmulQ4Fast(
    _ x: Tensor, weight: Tensor, scales: Tensor, biases: Tensor,
    M: Int, K: Int, N: Int, groupSize: Int
) -> Tensor {
    var gemmDesc = GEMMDescriptor()
    gemmDesc.matrixDimensions = (M: UInt32(M), N: UInt32(N), K: UInt32(K))
    gemmDesc.memoryPrecisions = (A: .FP32, B: .FP16, C: .FP32)
    gemmDesc.transposeState = (A: false, B: true)
    gemmDesc.quantizedB = true
    gemmDesc.groupSize = UInt32(groupSize)

    let (kernel, pipeline) = GEMMKernel.pipeline(for: gemmDesc)
    let out = Tensor.zeros([M, N])

    let ctx = MetalContext.shared
    ctx.run { enc in
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

/// Argmax over a float buffer (CPU — fine for vocab-sized vector).
public func argmax(_ x: Tensor) -> Int {
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

/// Temperature sampling with top-p (nucleus) filtering.
public func sample(_ x: Tensor, temperature: Float = 0.7, topP: Float = 0.9) -> Int {
    let ptr = x.buffer.contents().assumingMemoryBound(to: Float.self)
    let n = x.count

    // Apply temperature
    var logits = [Float](repeating: 0, count: n)
    for i in 0..<n { logits[i] = ptr[i] / temperature }

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
