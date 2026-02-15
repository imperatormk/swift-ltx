// KV Cache for autoregressive decoding.
// Pre-allocates Metal buffers for max sequence length.
// All copies (append + compact) use GPU kernels — zero CPU memcpy.

import Metal

public class KVCache {
    public let maxSeqLen: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let numLayers: Int

    // Per-layer K and V buffers: [numKVHeads, maxSeqLen, headDim]
    public var keyBuffers: [MTLBuffer]
    public var valueBuffers: [MTLBuffer]
    public private(set) var currentLen: Int = 0

    public init(numLayers: Int, numKVHeads: Int, headDim: Int, maxSeqLen: Int = 4096) {
        self.maxSeqLen = maxSeqLen
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.numLayers = numLayers

        let ctx = MetalContext.shared
        let layerBytes = numKVHeads * maxSeqLen * headDim * 2  // float16
        self.keyBuffers = (0..<numLayers).map { _ in ctx.makeBuffer(length: layerBytes) }
        self.valueBuffers = (0..<numLayers).map { _ in ctx.makeBuffer(length: layerBytes) }
    }

    /// Append new K, V for `seqLen` tokens starting at position `currentLen`. GPU dispatch, no CPU memcpy.
    /// k, v are [numKVHeads, seqLen, headDim].
    public func append(layer: Int, k: Tensor, v: Tensor, seqLen: Int = 1) {
        var hd = UInt32(headDim)
        var sl = UInt32(seqLen)
        var msl = UInt32(maxSeqLen)
        var sp = UInt32(currentLen)
        let pipe = KernelCache.shared.pipeline("kv_append")
        let totalThreadsY = numKVHeads * seqLen

        MetalContext.shared.run { enc in
            // Append K
            enc.setComputePipelineState(pipe)
            enc.setBuffer(k.buffer, offset: 0, index: 0)
            enc.setBuffer(self.keyBuffers[layer], offset: 0, index: 1)
            enc.setBytes(&hd, length: 4, index: 2)
            enc.setBytes(&sl, length: 4, index: 3)
            enc.setBytes(&msl, length: 4, index: 4)
            enc.setBytes(&sp, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: self.headDim, height: totalThreadsY, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(self.headDim, 256), height: 1, depth: 1))

            // Append V (same encoder, no barrier needed — different dst buffers)
            enc.setBuffer(v.buffer, offset: 0, index: 0)
            enc.setBuffer(self.valueBuffers[layer], offset: 0, index: 1)
            enc.dispatchThreads(MTLSize(width: self.headDim, height: totalThreadsY, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(self.headDim, 256), height: 1, depth: 1))
        }
    }

    /// Append K (already [nKVHeads, seqLen, headDim]) and V (still [seqLen, nKVHeads, headDim]).
    /// V is transposed + appended in one fused kernel. Saves one dispatch.
    public func appendFused(layer: Int, k: Tensor, vTransposed: Tensor, seqLen: Int = 1) {
        var hd = UInt32(headDim)
        var sl = UInt32(seqLen)
        var msl = UInt32(maxSeqLen)
        var sp = UInt32(currentLen)
        var nh = UInt32(numKVHeads)
        let totalThreadsY = numKVHeads * seqLen

        MetalContext.shared.run { enc in
            // Append K (already transposed by fused transpose_sh_rope)
            let pipe = KernelCache.shared.pipeline("kv_append")
            enc.setComputePipelineState(pipe)
            enc.setBuffer(k.buffer, offset: 0, index: 0)
            enc.setBuffer(self.keyBuffers[layer], offset: 0, index: 1)
            enc.setBytes(&hd, length: 4, index: 2)
            enc.setBytes(&sl, length: 4, index: 3)
            enc.setBytes(&msl, length: 4, index: 4)
            enc.setBytes(&sp, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: self.headDim, height: totalThreadsY, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(self.headDim, 256), height: 1, depth: 1))

            // Append V with fused transpose (reads [seqLen, nKVHeads, headDim])
            let pipeFused = KernelCache.shared.pipeline("kv_append_transposed")
            enc.setComputePipelineState(pipeFused)
            enc.setBuffer(vTransposed.buffer, offset: 0, index: 0)
            enc.setBuffer(self.valueBuffers[layer], offset: 0, index: 1)
            enc.setBytes(&hd, length: 4, index: 2)
            enc.setBytes(&sl, length: 4, index: 3)
            enc.setBytes(&msl, length: 4, index: 4)
            enc.setBytes(&sp, length: 4, index: 5)
            enc.setBytes(&nh, length: 4, index: 6)
            enc.dispatchThreads(MTLSize(width: self.headDim, height: totalThreadsY, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: min(self.headDim, 256), height: 1, depth: 1))
        }
    }

    /// Call after appending all layers for token(s).
    public func incrementPosition(by count: Int = 1) {
        currentLen += count
    }

    /// Raw K buffer for direct strided access by attention kernels.
    public func rawKeyBuffer(layer: Int) -> MTLBuffer {
        keyBuffers[layer]
    }

    /// Raw V buffer for direct strided access by attention kernels.
    public func rawValueBuffer(layer: Int) -> MTLBuffer {
        valueBuffers[layer]
    }

    public func reset() {
        currentLen = 0
    }
}
