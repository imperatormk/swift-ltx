// T5-XXL Encoder — pure Metal/Swift, 4-bit quantized, layer-by-layer loading.
// 24 layers, 64 heads, d_model=4096, d_kv=64, d_ff=10240.
// GeGLU FFN, RMSNorm, relative position bias (32 buckets × 64 heads).
// No 1/sqrt(d) scaling on attention (T5 quirk).

import Foundation
import Metal
import FlashAttention

// MARK: - T5 Configuration

public struct T5Config {
    public var vocabSize: Int = 32128
    public var dModel: Int = 4096
    public var numHeads: Int = 64
    public var dKV: Int = 64
    public var dFF: Int = 10240
    public var numLayers: Int = 24
    public var numBuckets: Int = 32
    public var maxDistance: Int = 128
    public var groupSize: Int = 64
    public var eps: Float = 1e-6
    public init() {}
}

// MARK: - T5 Layer Weights (loaded one at a time)

struct T5LayerWeights {
    // Self-attention: q, k, v, o — all quantized [dModel, dModel] → [4096, 512] int32 packed
    let qWeight: Tensor, qScales: Tensor, qBiases: Tensor
    let kWeight: Tensor, kScales: Tensor, kBiases: Tensor
    let vWeight: Tensor, vScales: Tensor, vBiases: Tensor
    let oWeight: Tensor, oScales: Tensor, oBiases: Tensor
    // Layer norms
    let selfAttnNormWeight: Tensor  // [dModel] f16
    let ffnNormWeight: Tensor       // [dModel] f16
    // FFN: wi_0 (gate), wi_1 (up), wo (down)
    let wi0Weight: Tensor, wi0Scales: Tensor, wi0Biases: Tensor  // [dFF, dModel/8]
    let wi1Weight: Tensor, wi1Scales: Tensor, wi1Biases: Tensor  // [dFF, dModel/8]
    let woWeight: Tensor, woScales: Tensor, woBiases: Tensor     // [dModel, dFF/8]
}

// MARK: - T5 Encoder

public class T5Encoder {
    public let config: T5Config
    private let file: SafeTensorsFile

    // Persistent weights (small, kept in memory)
    private let embeddingWeight: Tensor        // [vocabSize, dModel] f16
    private let finalNormWeight: Tensor        // [dModel] f16
    private let relAttnBiasWeight: Tensor      // [32, 64] f16

    // Pre-computed position bias (computed once per encode call)
    private var positionBiasCache: (seqLen: Int, buffer: MTLBuffer)? = nil

    public var log: ((String) -> Void)? = nil

    public init(url: URL, config: T5Config = T5Config()) throws {
        self.config = config
        self.file = try SafeTensorsFile(url: url)

        // Load small persistent weights
        self.embeddingWeight = Self.loadTensor(file: file, name: "shared.weight")
        self.finalNormWeight = Self.loadTensorAsF32(file: file, name: "encoder.final_layer_norm.weight")
        self.relAttnBiasWeight = Self.loadTensor(
            file: file, name: "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
    }

    /// Encode text tokens → embeddings [1, seqLen, 4096] f32.
    /// Loads/unloads layer weights one at a time to minimize memory.
    public func encode(tokenIds: [Int], maxLength: Int = 128) -> MTLBuffer {
        let config = self.config
        let dModel = config.dModel
        let pool = MetalContext.shared.bufferPool

        // Truncate + EOS, no padding (avoids pad tokens exploding through FFN)
        var ids = Array(tokenIds.prefix(maxLength - 1))
        if ids.last != 1 { ids.append(1) } // EOS
        let S = ids.count

        // Embedding lookup: [S] → [S, dModel] f16 → f32
        let embF16 = embeddingWeight.buffer.contents().assumingMemoryBound(to: UInt16.self)
        let vocabSize = embeddingWeight.shape[0]
        let hiddenBuf = pool.get(length: S * dModel * 4)
        let hiddenPtr = hiddenBuf.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<S {
            let dstOff = i * dModel
            let tokenId = min(ids[i], vocabSize - 1)
                let srcOff = tokenId * dModel
                for d in 0..<dModel {
                    hiddenPtr[dstOff + d] = Float(Float16(bitPattern: embF16[srcOff + d]))
                }
        }

        log?("T5 embedding done, seqLen=\(S) \(f32Stats(hiddenBuf, count: S * dModel))")

        // Compute relative position bias: [numHeads, S, S] f32
        let posBias = computePositionBias(seqLen: S)
        log?("T5 posBias: \(f32Stats(posBias, count: config.numHeads * S * S))")

        // Process layers
        var hidden = hiddenBuf
        for layerIdx in 0..<config.numLayers {
            let layerW = loadLayerWeights(layerIdx: layerIdx)
            debugLayer = false
            hidden = runLayer(hidden, weights: layerW, posBias: posBias, mask: nil, S: S)
            log?("T5 layer \(layerIdx) done \(f32Stats(hidden, count: S * dModel))")
            // layerW goes out of scope → ARC frees MTLBuffers
        }

        // Final RMSNorm
        hidden = rmsNormF32(hidden, weight: finalNormWeight, eps: config.eps, dim: dModel, rows: S)
        log?("T5 final output: \(f32Stats(hidden, count: S * dModel))")
        log?("T5 encode done")

        return hidden
    }

    // MARK: - Layer Forward

    private var debugLayer = true

    private func runLayer(_ hiddenF32: MTLBuffer, weights: T5LayerWeights, posBias: MTLBuffer, mask: MTLBuffer?, S: Int) -> MTLBuffer {
        let dModel = config.dModel
        let nHeads = config.numHeads
        let dKV = config.dKV
        let dFF = config.dFF
        let gs = config.groupSize
        let pool = MetalContext.shared.bufferPool
        let dbg = debugLayer

        // 1. Self-attention sub-layer
        // Pre-norm
        let normed = rmsNormF32(hiddenF32, weight: weights.selfAttnNormWeight, eps: config.eps, dim: dModel, rows: S)
        if dbg { log?("  rmsNorm: \(f32Stats(normed, count: S * dModel))") }

        // QKV projections: [S, dModel] f32 → f16 → Q4 matmul → [S, dModel] f16 → f32
        let normedF16 = castF32toF16(normed, count: S * dModel)
        if dbg { log?("  normedF16: \(f16Stats(normedF16))") }

        var qF16 = matmulQ4Fast(normedF16, weight: weights.qWeight, scales: weights.qScales,
                                biases: weights.qBiases, M: S, K: dModel, N: dModel, groupSize: gs)
        if dbg { log?("  Q(f16): \(f16Stats(qF16))") }
        let kF16 = matmulQ4Fast(normedF16, weight: weights.kWeight, scales: weights.kScales,
                                biases: weights.kBiases, M: S, K: dModel, N: dModel, groupSize: gs)
        if dbg { log?("  K(f16): \(f16Stats(kF16))") }
        let vF16 = matmulQ4Fast(normedF16, weight: weights.vWeight, scales: weights.vScales,
                                biases: weights.vBiases, M: S, K: dModel, N: dModel, groupSize: gs)
        if dbg { log?("  V(f16): \(f16Stats(vF16))") }

        // T5 has no 1/sqrt(d) scaling. Flash attention applies 1/sqrt(d) internally.
        // Pre-scale Q by sqrt(d) to cancel: Q' = Q * sqrt(d), then flash does Q'K^T/sqrt(d) = QK^T
        let sqrtD = Float(dKV).squareRoot()
        qF16 = scaleF16(qF16, scale: sqrtD, count: S * dModel)
        if dbg { log?("  Q*sqrt(d)(f16): \(f16Stats(qF16))") }

        // Cast to f32 for flash attention
        let qF32 = castF16toF32(qF16, count: S * dModel)
        let kF32 = castF16toF32(kF16, count: S * dModel)
        let vF32 = castF16toF32(vF16, count: S * dModel)

        // Transpose: [S, nHeads, dKV] → [nHeads, S, dKV]
        let qT = transposeSHF32(qF32, seqLen: S, nHeads: nHeads, headDim: dKV)
        let kT = transposeSHF32(kF32, seqLen: S, nHeads: nHeads, headDim: dKV)
        let vT = transposeSHF32(vF32, seqLen: S, nHeads: nHeads, headDim: dKV)

        // Flash attention with position bias
        let attnOut = flashAttentionBidirectionalF32WithBias(
            q: qT, k: kT, v: vT, bias: posBias, mask: mask,
            R: S, C: S, D: dKV, nHeads: nHeads)
        if dbg { log?("  attnOut(f32): \(f32Stats(attnOut, count: nHeads * S * dKV))") }

        // Transpose back: [nHeads, S, dKV] → [S, nHeads*dKV]
        let attnFlat = transposeHSF32(attnOut, seqLen: S, nHeads: nHeads, headDim: dKV)

        // Output projection: f32 → f16 → Q4 → f32
        let attnFlatF16 = castF32toF16(attnFlat, count: S * dModel)
        let oOutF32 = matmulQ4FastF32(attnFlatF16, weight: weights.oWeight, scales: weights.oScales,
                                      biases: weights.oBiases, M: S, K: dModel, N: dModel, groupSize: gs).buffer
        if dbg { log?("  oProj(f32): \(f32Stats(oOutF32, count: S * dModel))") }

        // Residual
        var hidden = elemAddF32OOP(hiddenF32, oOutF32, count: S * dModel)
        if dbg { log?("  afterAttn+res: \(f32Stats(hidden, count: S * dModel))") }

        // 2. FFN sub-layer
        let normed2 = rmsNormF32(hidden, weight: weights.ffnNormWeight, eps: config.eps, dim: dModel, rows: S)
        if dbg { log?("  ffnNorm: \(f32Stats(normed2, count: S * dModel))") }
        let normed2F16 = castF32toF16(normed2, count: S * dModel)

        // GeGLU: gate = gelu(wi_0(x)), up = wi_1(x), out = wo(gate * up)
        let gateF16 = matmulQ4Fast(normed2F16, weight: weights.wi0Weight, scales: weights.wi0Scales,
                                   biases: weights.wi0Biases, M: S, K: dModel, N: dFF, groupSize: gs)
        if dbg { log?("  gate(f16): \(f16Stats(gateF16))") }
        let upF16 = matmulQ4Fast(normed2F16, weight: weights.wi1Weight, scales: weights.wi1Scales,
                                 biases: weights.wi1Biases, M: S, K: dModel, N: dFF, groupSize: gs)
        if dbg { log?("  up(f16): \(f16Stats(upF16))") }

        // GELU(gate) * up in f16
        let geluGate = geluApproximateF16(gateF16, count: S * dFF)
        let gegluOut = elemMulF16(geluGate, upF16, count: S * dFF)
        if dbg { log?("  geglu(f16): \(f16Stats(gegluOut))") }

        // Down projection (f32 output to avoid f16 overflow for large values)
        let downF32 = matmulQ4FastF32(gegluOut, weight: weights.woWeight, scales: weights.woScales,
                                      biases: weights.woBiases, M: S, K: dFF, N: dModel, groupSize: gs).buffer
        if dbg { log?("  down(f32): \(f32Stats(downF32, count: S * dModel))") }

        // Residual
        hidden = elemAddF32OOP(hidden, downF32, count: S * dModel)
        if dbg { log?("  afterFFN+res: \(f32Stats(hidden, count: S * dModel))") }

        return hidden
    }

    // MARK: - Weight Loading (per-layer)

    private func loadLayerWeights(layerIdx: Int) -> T5LayerWeights {
        let p = "encoder.block.\(layerIdx)"

        func loadQ(_ suffix: String) -> (Tensor, Tensor, Tensor) {
            let w = Self.loadTensor(file: file, name: "\(p).\(suffix).weight")
            let s = Self.loadTensorEnsureF16(file: file, name: "\(p).\(suffix).scales")
            let b = Self.loadTensorEnsureF16(file: file, name: "\(p).\(suffix).biases")
            return (w, s, b)
        }

        let (qW, qS, qB) = loadQ("layer.0.SelfAttention.q")
        let (kW, kS, kB) = loadQ("layer.0.SelfAttention.k")
        let (vW, vS, vB) = loadQ("layer.0.SelfAttention.v")
        let (oW, oS, oB) = loadQ("layer.0.SelfAttention.o")
        let selfNorm = Self.loadTensorAsF32(file: file, name: "\(p).layer.0.layer_norm.weight")
        let ffnNorm = Self.loadTensorAsF32(file: file, name: "\(p).layer.1.layer_norm.weight")
        let (wi0W, wi0S, wi0B) = loadQ("layer.1.DenseReluDense.wi_0")
        let (wi1W, wi1S, wi1B) = loadQ("layer.1.DenseReluDense.wi_1")
        let (woW, woS, woB) = loadQ("layer.1.DenseReluDense.wo")

        return T5LayerWeights(
            qWeight: qW, qScales: qS, qBiases: qB,
            kWeight: kW, kScales: kS, kBiases: kB,
            vWeight: vW, vScales: vS, vBiases: vB,
            oWeight: oW, oScales: oS, oBiases: oB,
            selfAttnNormWeight: selfNorm,
            ffnNormWeight: ffnNorm,
            wi0Weight: wi0W, wi0Scales: wi0S, wi0Biases: wi0B,
            wi1Weight: wi1W, wi1Scales: wi1S, wi1Biases: wi1B,
            woWeight: woW, woScales: woS, woBiases: woB
        )
    }

    /// Load tensor, converting f32 → f16 if needed (wo scales/biases are f32 in this checkpoint).
    private static func loadTensorEnsureF16(file: SafeTensorsFile, name: String) -> Tensor {
        guard let info = file.tensors[name], let ptr = file.pointer(for: name) else {
            fatalError("T5 weight not found: \(name)")
        }
        if info.dtype == .float32 {
            let count = info.byteCount / 4
            let src = ptr.assumingMemoryBound(to: Float.self)
            let buf = MetalContext.shared.device.makeBuffer(length: count * 2, options: .storageModeShared)!
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<count { dst[i] = Float16(src[i]).bitPattern }
            return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
        }
        let buf = MetalContext.shared.device.makeBuffer(
            bytes: ptr, length: info.byteCount, options: .storageModeShared)!
        return Tensor(buffer: buf, shape: info.shape, dtype: info.dtype)
    }

    /// Load tensor as-is (raw bytes).
    private static func loadTensor(file: SafeTensorsFile, name: String) -> Tensor {
        guard let info = file.tensors[name], let ptr = file.pointer(for: name) else {
            fatalError("T5 weight not found: \(name)")
        }
        let buf = MetalContext.shared.device.makeBuffer(
            bytes: ptr, length: info.byteCount, options: .storageModeShared)!
        return Tensor(buffer: buf, shape: info.shape, dtype: info.dtype)
    }

    /// Load tensor and convert f16 → f32 (needed for norm weights used by f32 kernels).
    private static func loadTensorAsF32(file: SafeTensorsFile, name: String) -> Tensor {
        guard let info = file.tensors[name], let ptr = file.pointer(for: name) else {
            fatalError("T5 weight not found: \(name)")
        }
        if info.dtype == .float16 {
            let count = info.byteCount / 2
            let src = ptr.assumingMemoryBound(to: UInt16.self)
            let buf = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
            let dst = buf.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<count { dst[i] = Float(Float16(bitPattern: src[i])) }
            return Tensor(buffer: buf, shape: info.shape, dtype: .float32)
        }
        let buf = MetalContext.shared.device.makeBuffer(
            bytes: ptr, length: info.byteCount, options: .storageModeShared)!
        return Tensor(buffer: buf, shape: info.shape, dtype: info.dtype)
    }

    // MARK: - Relative Position Bias

    /// Compute T5 relative position bias: [numHeads, S, S] f32.
    /// The bias table is [32, 64] f16 (32 buckets × 64 heads).
    private func computePositionBias(seqLen: Int) -> MTLBuffer {
        if let cache = positionBiasCache, cache.seqLen == seqLen {
            return cache.buffer
        }

        let S = seqLen
        let numHeads = config.numHeads
        let numBuckets = config.numBuckets
        let maxDistance = config.maxDistance
        let halfBuckets = numBuckets / 2
        let maxExact = halfBuckets / 2

        // Read bias table: [32, 64] f16
        let tablePtr = relAttnBiasWeight.buffer.contents().assumingMemoryBound(to: UInt16.self)
        var table = [[Float]](repeating: [Float](repeating: 0, count: numHeads), count: numBuckets)
        for b in 0..<numBuckets {
            for h in 0..<numHeads {
                table[b][h] = Float(Float16(bitPattern: tablePtr[b * numHeads + h]))
            }
        }

        // Compute bucket indices and lookup
        // Output layout: [numHeads, S, S] — head-major for flash attention
        let outBuf = MetalContext.shared.device.makeBuffer(
            length: numHeads * S * S * 4, options: .storageModeShared)!
        let outPtr = outBuf.contents().assumingMemoryBound(to: Float.self)

        for q in 0..<S {
            for k in 0..<S {
                let relPos = k - q  // context_pos - memory_pos
                let n = -relPos
                let isNeg = n < 0
                let absN = abs(n)

                var bucket: Int
                if absN < maxExact {
                    bucket = absN
                } else {
                    let logRatio = Foundation.log(Float(absN) / Float(maxExact)) /
                                   Foundation.log(Float(maxDistance) / Float(maxExact))
                    bucket = maxExact + Int(logRatio * Float(halfBuckets - maxExact))
                    bucket = min(bucket, halfBuckets - 1)
                }
                if isNeg { bucket += halfBuckets }

                for h in 0..<numHeads {
                    outPtr[h * S * S + q * S + k] = table[bucket][h]
                }
            }
        }

        positionBiasCache = (seqLen: S, buffer: outBuf)
        return outBuf
    }
}

// MARK: - Flash Attention with Bias

/// Flash attention f32 with additive attention bias.
/// bias: [numHeads, R, C] f32 — added to attention scores before softmax.
public func flashAttentionBidirectionalF32WithBias(
    q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, bias: MTLBuffer,
    mask: MTLBuffer? = nil,
    R: Int, C: Int, D: Int, nHeads: Int
) -> MTLBuffer {
    var attDesc = AttentionDescriptor()
    attDesc.lowPrecisionInputs = false
    attDesc.lowPrecisionIntermediates = false
    attDesc.lowPrecisionOutputs = false
    attDesc.matrixDimensions = (row: UInt32(R), column: UInt32(C), head: UInt16(D))
    attDesc.transposeState = (Q: false, K: false, V: false, O: false)
    attDesc.causal = false
    attDesc.hasMask = (mask != nil)
    attDesc.hasAttnBias = true
    attDesc.biasBatchStride = 0  // same bias for all batches
    attDesc.biasHeadStride = UInt32(R * C)  // each head has its own [R, C] bias
    attDesc.biasRepeatCount = 0

    let (kernel, pipeline) = AttentionKernel.pipeline(for: attDesc, type: .forward)

    let pool = MetalContext.shared.bufferPool
    let oBytes = nHeads * R * D * 4
    let lBytes = nHeads * R * 4
    let bufO = pool.get(length: oBytes)
    let bufL = pool.get(length: lBytes)
    let dummy = pool.get(length: 16)
    memset(bufO.contents(), 0, oBytes)
    memset(bufL.contents(), 0, lBytes)
    memset(dummy.contents(), 0, 16)

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
        enc.setBuffer(mask ?? dummy, offset: 0, index: 10)  // mask
        enc.setBuffer(bias, offset: 0, index: 11)    // attention bias
        AttentionKernel.dispatch(
            encoder: enc, kernel: kernel, pipeline: pipeline,
            batchedParams: batchParams, parallelizationDimension: R,
            numHeads: nHeads, batchSize: 1)
    }
    return bufO
}

// MARK: - Helper ops needed for T5

/// Scale f16 tensor by a constant.
public func scaleF16(_ x: Tensor, scale: Float, count: Int) -> Tensor {
    let out = Tensor.empty([count], dtype: .float16)
    var s = Float16(scale)
    let pipe = KernelCache.shared.pipeline("scale_f16")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&s, length: 2, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Cast f32 → f16.
public func castF32toF16(_ buf: MTLBuffer, count: Int) -> Tensor {
    let out = Tensor.empty([count], dtype: .float16)
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

/// Cast f16 → f32.
public func castF16toF32(_ x: Tensor, count: Int) -> MTLBuffer {
    let out = MetalContext.shared.bufferPool.get(length: count * 4)
    let pipe = KernelCache.shared.pipeline("cast_f16_to_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Element-wise multiply f16.
public func elemMulF16(_ a: Tensor, _ b: Tensor, count: Int) -> Tensor {
    let out = Tensor.empty([count], dtype: .float16)
    let pipe = KernelCache.shared.pipeline("elem_mul_f16")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}


// MARK: - T5 Unigram Tokenizer

public struct T5Tokenizer {
    public let vocab: [(String, Float)]  // (token, log_prob)
    public let tokenToId: [String: Int]

    /// Load from HuggingFace tokenizer.json (google-t5/t5-base format).
    public init(url: URL) throws {
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let model = json["model"] as! [String: Any]
        let vocabList = model["vocab"] as! [[Any]]

        // Vocab is a flat list of [token, score] where index = token ID
        var vocab: [(String, Float)] = []
        var tokenToId: [String: Int] = [:]

        for (id, item) in vocabList.enumerated() {
            let token = item[0] as! String
            let score = (item[1] as! NSNumber).floatValue
            vocab.append((token, score))
            tokenToId[token] = id
        }

        self.vocab = vocab
        self.tokenToId = tokenToId
    }

    /// Encode text → token IDs using greedy longest-match (good enough for T5).
    /// T5 SentencePiece uses ▁ (U+2581) as word boundary.
    public func encode(_ text: String) -> [Int] {
        // Normalize: prepend ▁, replace spaces with ▁
        let normalized = "▁" + text.replacingOccurrences(of: " ", with: "▁")
        let chars = Array(normalized)
        let n = chars.count

        // Viterbi: find best segmentation
        // best[i] = (score, backpointer) for position i
        var bestScore = [Float](repeating: -Float.infinity, count: n + 1)
        var bestBack = [Int](repeating: 0, count: n + 1)
        bestScore[0] = 0

        for i in 0..<n {
            if bestScore[i] == -Float.infinity { continue }
            // Try all substrings starting at i
            for j in (i + 1)...n {
                let substr = String(chars[i..<j])
                if let id = tokenToId[substr] {
                    let score = bestScore[i] + vocab[id].1
                    if score > bestScore[j] {
                        bestScore[j] = score
                        bestBack[j] = i
                    }
                }
            }
        }

        // Backtrack
        var tokens: [Int] = []
        var pos = n
        while pos > 0 {
            let prev = bestBack[pos]
            let substr = String(chars[prev..<pos])
            if let id = tokenToId[substr] {
                tokens.append(id)
            } else {
                tokens.append(tokenToId["<unk>"]!)
            }
            pos = prev
        }
        tokens.reverse()

        // Append EOS (token id 1)
        tokens.append(1)
        return tokens
    }
}
