// KV Cache for autoregressive decoding.
// Pre-allocates Metal buffers for max sequence length.

import Metal

public class KVCache {
    public let maxSeqLen: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let numLayers: Int

    // Per-layer K and V buffers: [numKVHeads, maxSeqLen, headDim]
    public var keyBuffers: [MTLBuffer]
    public var valueBuffers: [MTLBuffer]
    // Reusable compacted buffers (2 per layer: K and V)
    private var compactKeyBuffers: [MTLBuffer]
    private var compactValueBuffers: [MTLBuffer]
    public private(set) var currentLen: Int = 0

    public init(numLayers: Int, numKVHeads: Int, headDim: Int, maxSeqLen: Int = 4096) {
        self.maxSeqLen = maxSeqLen
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.numLayers = numLayers

        let ctx = MetalContext.shared
        let layerBytes = numKVHeads * maxSeqLen * headDim * 4  // float32
        self.keyBuffers = (0..<numLayers).map { _ in ctx.makeBuffer(length: layerBytes) }
        self.valueBuffers = (0..<numLayers).map { _ in ctx.makeBuffer(length: layerBytes) }
        self.compactKeyBuffers = (0..<numLayers).map { _ in ctx.makeBuffer(length: layerBytes) }
        self.compactValueBuffers = (0..<numLayers).map { _ in ctx.makeBuffer(length: layerBytes) }
    }

    /// Append new K, V for `seqLen` tokens starting at position `currentLen`.
    /// k, v are [numKVHeads, seqLen, headDim].
    public func append(layer: Int, k: Tensor, v: Tensor, seqLen: Int = 1) {
        let headBytes = headDim * 4
        let seqStride = maxSeqLen * headBytes
        let pos = currentLen

        let kSrc = k.buffer.contents()
        let vSrc = v.buffer.contents()
        let kDst = keyBuffers[layer].contents()
        let vDst = valueBuffers[layer].contents()

        for h in 0..<numKVHeads {
            let srcOff = h * seqLen * headBytes
            let dstOff = h * seqStride + pos * headBytes
            memcpy(kDst + dstOff, kSrc + srcOff, seqLen * headBytes)
            memcpy(vDst + dstOff, vSrc + srcOff, seqLen * headBytes)
        }
    }

    /// Call after appending all layers for token(s).
    public func incrementPosition(by count: Int = 1) {
        currentLen += count
    }

    /// Get K tensor for attention: [numKVHeads, seqLen, headDim] (compacted, reuses buffer)
    public func keys(layer: Int, seqLen: Int) -> Tensor {
        return compacted(keyBuffers[layer], into: compactKeyBuffers[layer], seqLen: seqLen)
    }

    /// Get V tensor for attention: [numKVHeads, seqLen, headDim] (compacted, reuses buffer)
    public func values(layer: Int, seqLen: Int) -> Tensor {
        return compacted(valueBuffers[layer], into: compactValueBuffers[layer], seqLen: seqLen)
    }

    /// Copy strided [numKVHeads, maxSeqLen, headDim] → contiguous [numKVHeads, seqLen, headDim]
    private func compacted(_ buf: MTLBuffer, into out: MTLBuffer, seqLen: Int) -> Tensor {
        let headBytes = headDim * 4
        let srcStride = maxSeqLen * headBytes
        let dstStride = seqLen * headBytes
        let src = buf.contents()
        let dst = out.contents()
        for h in 0..<numKVHeads {
            memcpy(dst + h * dstStride, src + h * srcStride, seqLen * headBytes)
        }
        return Tensor(buffer: out, shape: [numKVHeads, seqLen, headDim])
    }

    public func reset() {
        currentLen = 0
    }
}
