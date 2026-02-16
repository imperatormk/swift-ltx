// Weight loading from safetensors into Metal buffers.
// Supports both float (f32/f16/bf16) and MLX 4-bit quantized weights.

import Metal

/// A linear projection — either float or 4-bit quantized.
public enum LinearWeight {
    case float(Tensor)                                    // [N, K] float32
    case quantized(weight: Tensor, scales: Tensor, biases: Tensor, K: Int, groupSize: Int)
    // weight: [N, K/8] uint32, scales/biases: [N, K/groupSize] float16
}

public struct LayerWeights {
    // Attention
    public let qProj: LinearWeight
    public let kProj: LinearWeight
    public let vProj: LinearWeight
    public let oProj: LinearWeight

    // FFN (SwiGLU)
    public let gateProj: LinearWeight
    public let upProj: LinearWeight
    public let downProj: LinearWeight

    // Norms (always float)
    public let inputNormWeight: Tensor
    public let postNormWeight: Tensor

    public let numAttentionHeads: Int
    public let intermediateSize: Int
}

public struct ModelWeights {
    public let embedTokens: LinearWeight  // [vocabSize, hidden] — may be quantized
    public let normWeight: Tensor      // [hidden]
    public let lmHead: LinearWeight    // [vocabSize, hidden] — may be quantized or tied
    public let layers: [LayerWeights]
    public let isQuantized: Bool

    public init(config: ModelConfig, files: [SafeTensorsFile]) throws {
        let ctx = MetalContext.shared

        // Check if model is quantized (MLX style: .weight is uint32 + .scales/.biases exist)
        let firstQProj = "model.layers.0.self_attn.q_proj.weight"
        let quantized = files.contains { file in
            if let info = file.tensors[firstQProj] {
                return info.dtype == .uint8 || info.dtype == .int8 ||
                       file.tensors["model.layers.0.self_attn.q_proj.scales"] != nil
            }
            return false
        }
        self.isQuantized = quantized

        func exists(_ name: String) -> Bool {
            files.contains { $0.tensors[name] != nil }
        }

        func loadRaw(_ name: String) -> Tensor {
            for file in files {
                if let info = file.tensors[name] {
                    let ptr = file.pointer(for: name)!
                    let buf: MTLBuffer
                    switch info.dtype {
                    case .float32:
                        buf = ctx.device.makeBuffer(
                            bytes: ptr, length: info.byteCount, options: .storageModeShared)!
                    case .float16:
                        // Keep as f16 for quantized scales/biases, convert for float weights
                        buf = ctx.device.makeBuffer(
                            bytes: ptr, length: info.byteCount, options: .storageModeShared)!
                    case .bfloat16:
                        let count = info.byteCount / 2
                        let src = ptr.assumingMemoryBound(to: UInt16.self)
                        var f32 = [Float](repeating: 0, count: count)
                        for i in 0..<count { f32[i] = bfloat16ToFloat32(src[i]) }
                        buf = f32.withUnsafeBytes {
                            ctx.device.makeBuffer(bytes: $0.baseAddress!, length: count * 4, options: .storageModeShared)!
                        }
                        return Tensor(buffer: buf, shape: info.shape, dtype: .float32)
                    case .uint8, .int8:
                        // Packed quantized weights — load raw
                        buf = ctx.device.makeBuffer(
                            bytes: ptr, length: info.byteCount, options: .storageModeShared)!
                    case .int32, .uint32:
                        buf = ctx.device.makeBuffer(
                            bytes: ptr, length: info.byteCount, options: .storageModeShared)!
                    }
                    return Tensor(buffer: buf, shape: info.shape, dtype: info.dtype)
                }
            }
            fatalError("Weight not found: \(name)")
        }

        /// Load as float32 (converting f16 if needed).
        func loadFloat(_ name: String) -> Tensor {
            let t = loadRaw(name)
            if t.dtype == .float16 {
                // Convert f16 → f32
                let count = t.byteCount / 2
                let src = t.buffer.contents().assumingMemoryBound(to: UInt16.self)
                var f32 = [Float](repeating: 0, count: count)
                for i in 0..<count { f32[i] = float16ToFloat32(src[i]) }
                return Tensor(f32, shape: t.shape)
            }
            return t
        }

        /// Load a linear weight (auto-detect quantized vs float).
        func loadLinear(_ prefix: String, K: Int) -> LinearWeight {
            if exists("\(prefix).scales") {
                // Quantized: weight (uint32 packed), scales (f16), biases (f16)
                let w = loadRaw("\(prefix).weight")
                let s = loadRaw("\(prefix).scales")
                let b = loadRaw("\(prefix).biases")
                let groupSize = config.quantization?.groupSize ?? 64
                return .quantized(weight: w, scales: s, biases: b, K: K, groupSize: groupSize)
            } else {
                return .float(loadFloat("\(prefix).weight"))
            }
        }

        self.embedTokens = loadLinear("model.embed_tokens", K: config.hiddenSize)
        self.normWeight = loadFloat("model.norm.weight")

        // lm_head
        if exists("lm_head.weight") {
            self.lmHead = loadLinear("lm_head", K: config.hiddenSize)
        } else {
            self.lmHead = self.embedTokens  // tied
        }

        var layers: [LayerWeights] = []
        for i in 0..<config.numHiddenLayers {
            let p = "model.layers.\(i)"
            let h = config.hiddenSize
            let qDim = config.numAttentionHeads * config.headDim
            let kvDim = config.numKeyValueHeads * config.headDim

            layers.append(LayerWeights(
                qProj: loadLinear("\(p).self_attn.q_proj", K: h),
                kProj: loadLinear("\(p).self_attn.k_proj", K: h),
                vProj: loadLinear("\(p).self_attn.v_proj", K: h),
                oProj: loadLinear("\(p).self_attn.o_proj", K: qDim),
                gateProj: loadLinear("\(p).mlp.gate_proj", K: h),
                upProj: loadLinear("\(p).mlp.up_proj", K: h),
                downProj: loadLinear("\(p).mlp.down_proj", K: config.intermediateSize),
                inputNormWeight: loadFloat("\(p).input_layernorm.weight"),
                postNormWeight: loadFloat("\(p).post_attention_layernorm.weight"),
                numAttentionHeads: config.numAttentionHeads,
                intermediateSize: config.intermediateSize
            ))
        }
        self.layers = layers
    }
}

// MARK: - Float conversion helpers

private func float16ToFloat32(_ h: UInt16) -> Float {
    let sign = UInt32(h >> 15) << 31
    let exp = UInt32((h >> 10) & 0x1F)
    let frac = UInt32(h & 0x3FF)

    if exp == 0 {
        if frac == 0 { return Float(bitPattern: sign) }
        var f = frac
        var e: UInt32 = 0
        while f & 0x400 == 0 { f <<= 1; e += 1 }
        let bits = sign | ((127 - 15 + 1 - e) << 23) | ((f & 0x3FF) << 13)
        return Float(bitPattern: bits)
    } else if exp == 31 {
        let bits = sign | 0x7F800000 | (frac << 13)
        return Float(bitPattern: bits)
    }

    let bits = sign | ((exp + 112) << 23) | (frac << 13)
    return Float(bitPattern: bits)
}

func bfloat16ToFloat32(_ b: UInt16) -> Float {
    Float(bitPattern: UInt32(b) << 16)
}
