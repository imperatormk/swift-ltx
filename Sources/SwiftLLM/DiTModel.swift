// LTX-Video DiT (Diffusion Transformer) — pure Swift + Metal.
// Port of the MLX-Swift LTXVideo library.
// No MLX dependency — uses our Tensor, Metal kernels, and FlashAttention IR.

import Metal
import FlashAttention
import MetalASM

// MARK: - Configuration

public struct LTXTransformerConfig {
    public var numAttentionHeads: Int = 32
    public var attentionHeadDim: Int = 64
    public var inChannels: Int = 128
    public var outChannels: Int = 128
    public var numLayers: Int = 28
    public var crossAttentionDim: Int = 2048
    public var captionChannels: Int = 4096
    public var activationFn: String = "gelu-approximate"
    public var attentionBias: Bool = true
    public var normEps: Float = 1e-6
    public var qkNorm: Bool = true  // rms_norm on Q/K
    public var adaptiveNorm: String = "single_scale_shift"  // or "single_scale"
    public var positionalEmbeddingTheta: Float = 10000.0
    public var positionalEmbeddingMaxPos: [Int] = [20, 2048, 2048]
    public var timestepScaleMultiplier: Float = 1000.0
    public var patchSize: Int = 1

    public var innerDim: Int { numAttentionHeads * attentionHeadDim }

    public init() {}
}

public struct RectifiedFlowConfig {
    public var numTrainTimesteps: Int = 1000
    public var shifting: String? = nil  // nil or "SD3"
    public var targetShiftTerminal: Float = 0.1
    public var sampler: String = "LinearQuadratic"

    public init() {}
}

// MARK: - DiT Weights

/// Linear layer weights: weight [N, K] (bf16 or f32), bias [N] f16 (optional), biasF32 [N] f32 (optional).
public struct DiTLinear {
    public let weight: Tensor    // [N, K] bf16/f16/f32, or [N, K/8] int32 (NF4 packed)
    public let bias: Tensor?     // [N] f16
    public let biasF32: MTLBuffer?  // [N] f32 (for all-f32 path)
    public let K: Int
    public let N: Int
    // NF4 quantized fields (nil for dense weights)
    public let q4Scales: Tensor?   // [N, K/groupSize] f16
    public let q4Biases: Tensor?   // [N, K/groupSize] f16
    public let q4GroupSize: Int

    public init(weight: Tensor, bias: Tensor?, K: Int, N: Int,
                q4Scales: Tensor? = nil, q4Biases: Tensor? = nil, q4GroupSize: Int = 64) {
        self.weight = weight
        self.bias = bias
        self.K = K
        self.N = N
        self.q4Scales = q4Scales
        self.q4Biases = q4Biases
        self.q4GroupSize = q4GroupSize
        // Pre-convert bias to f32
        if let b = bias {
            let buf = MetalContext.shared.device.makeBuffer(length: N * 4, options: .storageModeShared)!
            let src = b.buffer.contents().assumingMemoryBound(to: Float16.self)
            let dst = buf.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<N { dst[i] = Float(src[i]) }
            self.biasF32 = buf
        } else {
            self.biasF32 = nil
        }
    }

    public var isQ4: Bool { q4Scales != nil }

    /// out = x @ W^T + bias. x: [M, K] f16, out: [M, N] f16. All compute in f32.
    public func apply(_ x: Tensor, M: Int) -> Tensor {
        let f32buf = applyF32(x, M: M)
        return castF32toF16(f32buf, count: M * N, shape: [M, N])
    }

    /// f16 input → f32, × weight → f32 output.
    public func applyF32(_ x: Tensor, M: Int) -> MTLBuffer {
        let xF32 = castF16toF32(x)
        return applyF32in(xF32, M: M)
    }

    /// f16 input × f16 weight → f16 output (Tensor).
    public func applyF16(_ x: Tensor, M: Int) -> Tensor {
        let out = simpleMatmulF16(x, weight, M: M, K: K, N: N)
        if let bias {
            addBiasF16(out, bias: bias, M: M, N: N)
        }
        return out
    }

    /// f32 input × weight → f32 output. Weight can be f16, bf16, f32, or NF4.
    public func applyF32in(_ x: MTLBuffer, M: Int) -> MTLBuffer {
        let f32buf: MTLBuffer
        if let scales = q4Scales, let biases = q4Biases {
            // NF4 path: x is f32, cast to f16 for matmulQ4FastF32
            let xF16 = castF32toF16(x, count: M * K, shape: [M, K])
            f32buf = matmulQ4FastF32(
                xF16,
                weight: weight, scales: scales, biases: biases,
                M: M, K: K, N: N, groupSize: q4GroupSize).buffer
        } else {
            switch weight.dtype {
            case .float16:
                f32buf = matmulF32xF16(x, weight, M: M, K: K, N: N)
            case .bfloat16:
                f32buf = matmulF32xBF16(x, weight.buffer, M: M, K: K, N: N)
            default:
                f32buf = matmulF32xF32(x, weight.buffer, M: M, K: K, N: N)
            }
        }
        if let biasF32 {
            addBiasF32F32(f32buf, bias: biasF32, M: M, N: N)
        }
        return f32buf
    }
}

/// Broadcast [N] bias to [M, N] by repeating M times.
func broadcastBias(_ bias: Tensor, M: Int, N: Int) -> Tensor {
    if M == 1 { return bias }
    // Use a simple kernel or just tile on CPU — for now use repeat via GPU copy
    let out = Tensor.empty([M, N], dtype: .float16)
    let pipe = KernelCache.shared.pipeline("broadcast_bias")
    var n = UInt32(N)
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(bias.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: M * N, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

public struct DiTLayerNormWeights {
    public let weight: Tensor  // [dim] f16
    public let bias: Tensor    // [dim] f16
    public let weightF32: MTLBuffer  // [dim] f32
    public let biasF32: MTLBuffer    // [dim] f32

    public init(weight: Tensor, bias: Tensor) {
        self.weight = weight
        self.bias = bias
        let dim = weight.count
        let ctx = MetalContext.shared
        // Convert f16 weight/bias to f32
        let wBuf = ctx.device.makeBuffer(length: dim * 4, options: .storageModeShared)!
        let bBuf = ctx.device.makeBuffer(length: dim * 4, options: .storageModeShared)!
        let wSrc = weight.buffer.contents().assumingMemoryBound(to: Float16.self)
        let wDst = wBuf.contents().assumingMemoryBound(to: Float.self)
        let bSrc = bias.buffer.contents().assumingMemoryBound(to: Float16.self)
        let bDst = bBuf.contents().assumingMemoryBound(to: Float.self)
        for i in 0..<dim { wDst[i] = Float(wSrc[i]); bDst[i] = Float(bSrc[i]) }
        self.weightF32 = wBuf
        self.biasF32 = bBuf
    }

    public func apply(_ x: Tensor, dim: Int) -> Tensor {
        layerNorm(x, weight: weight, bias: bias, dim: dim)
    }

    public func applyF32in(_ x: MTLBuffer, dim: Int, rows: Int) -> Tensor {
        layerNormF32in(x, weight: weight, bias: bias, dim: dim, rows: rows)
    }

    /// f32 input → f32 output layer norm
    public func applyF32(_ x: MTLBuffer, dim: Int, rows: Int) -> MTLBuffer {
        layerNormF32(x, weight: weightF32, bias: biasF32, dim: dim, rows: rows)
    }
}

public struct DiTRMSNormWeights {
    public let weight: Tensor  // [dim] f32

    public func apply(_ x: Tensor, dim: Int, eps: Float = 1e-6) -> Tensor {
        rmsNorm(x, weight: weight, eps: eps, dim: dim)
    }

    /// Apply RMS norm with f32 input buffer → f16 output. No precision loss.
    public func applyF32in(_ x: MTLBuffer, dim: Int, rows: Int, eps: Float = 1e-6) -> Tensor {
        rmsNormF32in(x, weight: weight, eps: eps, dim: dim, rows: rows)
    }

    /// Apply RMS norm with f32 input → f32 output.
    public func applyF32(_ x: MTLBuffer, dim: Int, rows: Int, eps: Float = 1e-6) -> MTLBuffer {
        rmsNormF32(x, weight: weight, eps: eps, dim: dim, rows: rows)
    }
}

public struct DiTAttentionWeights {
    public let toQ: DiTLinear
    public let toK: DiTLinear
    public let toV: DiTLinear
    public let toOut: DiTLinear
    public let qNorm: DiTRMSNormWeights?
    public let kNorm: DiTRMSNormWeights?
    public let useRope: Bool
    public let isCrossAttention: Bool
}

public struct DiTFFNWeights {
    public let proj: DiTLinear      // GEGLU: [dim, innerDim*2], or GELU: [dim, innerDim]
    public let projOut: DiTLinear   // [innerDim, dim]
    public let isGeglu: Bool
}

public struct DiTBlockWeights {
    public let norm1: DiTRMSNormWeights
    public let attn1: DiTAttentionWeights   // self-attention
    public let norm2: DiTRMSNormWeights
    public let attn2: DiTAttentionWeights?  // cross-attention
    public let ff: DiTFFNWeights
    public let scaleShiftTable: Tensor      // [numAdaParams, dim] f16
    public let numAdaParams: Int            // 4 (single_scale) or 6 (single_scale_shift)
}

public struct DiTWeights {
    public let patchifyProj: DiTLinear              // [inChannels, innerDim]
    public let adalnTimestepMLP1: DiTLinear          // [256, dim]
    public let adalnTimestepMLP2: DiTLinear          // [dim, dim]
    public let adalnLinear: DiTLinear                // [dim, numParams*dim]
    public let captionProj1: DiTLinear?              // [4096, dim]
    public let captionProj2: DiTLinear?              // [dim, dim]
    public let blockCount: Int
    public let loadBlock: (Int) -> DiTBlockWeights   // lazy: loads from mmap on demand
    public let normOut: DiTLayerNormWeights
    public let scaleShiftTable: Tensor               // [2, dim] f16
    public let projOut: DiTLinear                    // [dim, outChannels]
    public let config: LTXTransformerConfig
}

// MARK: - 3D RoPE

/// Precompute 3D RoPE cos/sin frequency tensors on CPU, return as f32 Metal buffers.
/// indicesGrid: [3, numTokens] as Float array (frame, height, width coords).
/// Returns (cosFreqs, sinFreqs) each [numTokens, dim] as f32 MTLBuffer.
public func precomputeFreqsCIS3D(
    indicesGrid: [[Float]],  // [3][numTokens]
    maxPos: [Int],
    theta: Float,
    dim: Int
) -> (MTLBuffer, MTLBuffer) {
    let numTokens = indicesGrid[0].count
    let numFreqs = dim / 6

    // Compute scaled indices: theta^(linspace(0, 1, numFreqs)) * pi/2
    var scaledIndices = [Float](repeating: 0, count: numFreqs)
    for i in 0..<numFreqs {
        let exponent = Float(i) / Float(max(numFreqs - 1, 1))
        scaledIndices[i] = powf(theta, exponent) * (.pi / 2.0)
    }

    let freqsDim = numFreqs * 3  // dim/2
    var cosData = [Float](repeating: 0, count: numTokens * dim)
    var sinData = [Float](repeating: 0, count: numTokens * dim)

    for t in 0..<numTokens {
        var fracs = [Float](repeating: 0, count: 3)
        for d in 0..<3 {
            fracs[d] = indicesGrid[d][t] / Float(maxPos[d])
        }

        var flatFreqs = [Float](repeating: 0, count: freqsDim)
        for fi in 0..<numFreqs {
            for di in 0..<3 {
                let fracScaled = fracs[di] * 2.0 - 1.0
                flatFreqs[fi * 3 + di] = fracScaled * scaledIndices[fi]
            }
        }

        let actualDim = freqsDim * 2
        let padSize = dim - actualDim

        let base = t * dim
        for i in 0..<padSize {
            cosData[base + i] = 1.0
            sinData[base + i] = 0.0
        }
        for i in 0..<freqsDim {
            let c = cosf(flatFreqs[i])
            let s = sinf(flatFreqs[i])
            cosData[base + padSize + i * 2] = c
            cosData[base + padSize + i * 2 + 1] = c
            sinData[base + padSize + i * 2] = s
            sinData[base + padSize + i * 2 + 1] = s
        }
    }

    let ctx = MetalContext.shared
    let cosB = cosData.withUnsafeBytes { ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
    let sinB = sinData.withUnsafeBytes { ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
    return (cosB, sinB)
}

// MARK: - Scheduler

public class RectifiedFlowScheduler {
    public let config: RectifiedFlowConfig
    public var timesteps: [Float] = []
    public var numInferenceSteps: Int = 0

    public init(config: RectifiedFlowConfig = RectifiedFlowConfig()) {
        self.config = config
    }

    public func setTimesteps(numSteps: Int, numTokens: Int) {
        if config.sampler == "LinearQuadratic" {
            timesteps = linearQuadraticSchedule(numSteps: numSteps)
        } else {
            timesteps = (0..<numSteps).map { Float(1) - Float($0) / Float(numSteps) }
        }

        if config.shifting == "SD3" {
            timesteps = sd3TimestepShift(numTokens: numTokens, timesteps: timesteps)
            timesteps = stretchShiftsToTerminal(timesteps, terminal: config.targetShiftTerminal)
        }
        self.numInferenceSteps = numSteps
    }

    /// One denoising step: x_{t-1} = x_t - dt * v_pred
    public func step(modelOutput: Tensor, timestepIndex: Int, sample: Tensor) -> Tensor {
        let t = timesteps[timestepIndex]
        let tNext = timestepIndex + 1 < timesteps.count ? timesteps[timestepIndex + 1] : Float(0)
        let dt = t - tNext

        // sample - dt * modelOutput
        // Use element-wise: out = sample + (-dt) * modelOutput
        return elemAddScaled(sample, modelOutput, scale: -dt)
    }

    /// One denoising step (f32): x_{t-1} = x_t - dt * v_pred
    public func stepF32(modelOutput: MTLBuffer, timestepIndex: Int, sample: MTLBuffer, count: Int) -> MTLBuffer {
        let t = timesteps[timestepIndex]
        let tNext = timestepIndex + 1 < timesteps.count ? timesteps[timestepIndex + 1] : Float(0)
        let dt = t - tNext
        return elemAddScaledF32(sample, modelOutput, scale: -dt, count: count)
    }

    private func linearQuadraticSchedule(numSteps: Int, thresholdNoise: Float = 0.025) -> [Float] {
        if numSteps == 1 { return [1.0] }
        let linearSteps = numSteps / 2
        var sigmas = (0..<linearSteps).map { Float($0) * thresholdNoise / Float(linearSteps) }
        let thresholdNoiseDiff = Float(linearSteps) - thresholdNoise * Float(numSteps)
        let quadSteps = numSteps - linearSteps
        let quadCoef = thresholdNoiseDiff / (Float(linearSteps) * Float(quadSteps * quadSteps))
        let linCoef = thresholdNoise / Float(linearSteps) - 2.0 * thresholdNoiseDiff / Float(quadSteps * quadSteps)
        let c = quadCoef * Float(linearSteps * linearSteps)
        for i in linearSteps..<numSteps {
            sigmas.append(quadCoef * Float(i * i) + linCoef * Float(i) + c)
        }
        sigmas.append(1.0)
        return sigmas.dropLast().map { 1.0 - $0 }
    }

    private func sd3TimestepShift(numTokens: Int, timesteps: [Float]) -> [Float] {
        let minTokens = 1024, maxTokens = 4096
        let minShift: Float = 0.95, maxShift: Float = 2.05
        let m = (maxShift - minShift) / Float(maxTokens - minTokens)
        let b = minShift - m * Float(minTokens)
        let mu = m * Float(numTokens) + b
        let expMu = expf(mu)
        return timesteps.map { t in
            expMu / (expMu + powf(1.0 / t - 1.0, 1.0))
        }
    }

    private func stretchShiftsToTerminal(_ shifts: [Float], terminal: Float) -> [Float] {
        let lastVal = 1.0 - shifts.last!
        let scaleFactor = lastVal / (1.0 - terminal)
        return shifts.map { 1.0 - (1.0 - $0) / scaleFactor }
    }
}

// MARK: - Patchifier

public struct SymmetricPatchifier {
    public let patchSize: Int

    public init(patchSize: Int = 1) {
        self.patchSize = patchSize
    }

    /// Patchify: [B, C, F, H, W] → [B, numTokens, C*patchSize^2]
    /// Also returns coordinate grid [3][numTokens] for RoPE.
    /// Since patchSize=1 for LTX 0.9.8, this is just a reshape:
    /// [1, C, F, H, W] → [1, F*H*W, C]
    public func patchify(_ latents: Tensor, B: Int, C: Int, F: Int, H: Int, W: Int) -> (Tensor, [[Float]]) {
        let pH = H / patchSize
        let pW = W / patchSize
        let numTokens = F * pH * pW
        let tokenDim = C * patchSize * patchSize

        // For patchSize=1, this is just transpose from NCFHW to N(FHW)C
        // We need a GPU kernel for the general case, but for patchSize=1:
        let tokens: Tensor
        if patchSize == 1 {
            // [B, C, F, H, W] channel-first → [B, F*H*W, C] channel-last
            // Each token t = f*H*W + h*W + w maps to input[:, :, f, h, w]
            tokens = transposeNCtoNLC(latents, B: B, C: C, spatial: F * H * W)
        } else {
            // General case with patchSize > 1
            tokens = patchifyGeneral(latents, B: B, C: C, F: F, H: H, W: W)
        }

        // Coordinate grid
        let coords = getLatentCoords(frames: F, height: H, width: W)

        return (tokens, coords)
    }

    /// Unpatchify: [B, numTokens, tokenDim] → [B, C, F, H, W]
    public func unpatchify(_ tokens: Tensor, B: Int, C: Int, F: Int, H: Int, W: Int) -> Tensor {
        if patchSize == 1 {
            // [B, F*H*W, C] → [B, C, F*H*W] → reshape to [B, C, F, H, W]
            let flat = transposeNLCtoNC(tokens, B: B, C: C, spatial: F * H * W)
            return Tensor(buffer: flat.buffer, shape: [B, C, F, H, W], dtype: flat.dtype)
        }
        return unpatchifyGeneral(tokens, B: B, C: C, F: F, H: H, W: W)
    }

    /// Patchify f32: [B, C, F, H, W] f32 MTLBuffer → [B, numTokens, C] f32 MTLBuffer + coords
    public func patchifyF32(_ latents: MTLBuffer, B: Int, C: Int, F: Int, H: Int, W: Int) -> (MTLBuffer, [[Float]]) {
        precondition(patchSize == 1, "patchifyF32 only supports patchSize=1")
        let tokens = transposeNCtoNLCF32(latents, B: B, C: C, spatial: F * H * W)
        let coords = getLatentCoords(frames: F, height: H, width: W)
        return (tokens, coords)
    }

    /// Unpatchify f32: [B, numTokens, C] f32 MTLBuffer → [B, C, F, H, W] f32 MTLBuffer
    public func unpatchifyF32(_ tokens: MTLBuffer, B: Int, C: Int, F: Int, H: Int, W: Int) -> MTLBuffer {
        precondition(patchSize == 1, "unpatchifyF32 only supports patchSize=1")
        return transposeNLCtoNCF32(tokens, B: B, C: C, spatial: F * H * W)
    }

    func getLatentCoords(frames: Int, height: Int, width: Int) -> [[Float]] {
        let pH = height / patchSize
        let pW = width / patchSize
        var fCoords = [Float]()
        var hCoords = [Float]()
        var wCoords = [Float]()
        for f in 0..<frames {
            for h in stride(from: 0, to: height, by: patchSize) {
                for w in stride(from: 0, to: width, by: patchSize) {
                    fCoords.append(Float(f))
                    hCoords.append(Float(h))
                    wCoords.append(Float(w))
                }
            }
        }
        return [fCoords, hCoords, wCoords]
    }
}

/// Transpose [B, C, S] → [B, S, C] (channel-first to channel-last)
func transposeNCtoNLC(_ x: Tensor, B: Int, C: Int, spatial: Int) -> Tensor {
    let out = Tensor.empty([B, spatial, C], dtype: .float16)
    // Reuse transpose_sh: treats as [S, C, 1] → ... no, let's just add a kernel
    // Actually transpose_sh does [seqLen, nHeads, headDim] → [nHeads, seqLen, headDim]
    // We want [C, spatial] → [spatial, C] per batch = transpose_sh with seqLen=C, nHeads=spatial, headDim=1
    // That's wasteful. Use a simple copy kernel.
    var c = UInt32(C), s = UInt32(spatial)
    let pipe = KernelCache.shared.pipeline("transpose_cs")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&c, length: 4, index: 2)
        enc.setBytes(&s, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: C * spatial, height: B, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Transpose [B, S, C] → [B, C, S] (channel-last to channel-first)
func transposeNLCtoNC(_ x: Tensor, B: Int, C: Int, spatial: Int) -> Tensor {
    let out = Tensor.empty([B, C, spatial], dtype: .float16)
    var c = UInt32(C), s = UInt32(spatial)
    let pipe = KernelCache.shared.pipeline("transpose_sc")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&c, length: 4, index: 2)
        enc.setBytes(&s, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: C * spatial, height: B, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Transpose [B, C, S] → [B, S, C] f32
func transposeNCtoNLCF32(_ x: MTLBuffer, B: Int, C: Int, spatial: Int) -> MTLBuffer {
    let count = B * C * spatial
    let out = MetalContext.shared.bufferPool.get(length: count * 4)
    var c = UInt32(C), s = UInt32(spatial)
    let pipe = KernelCache.shared.pipeline("transpose_cs_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&c, length: 4, index: 2)
        enc.setBytes(&s, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: C * spatial, height: B, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Transpose [B, S, C] → [B, C, S] f32
func transposeNLCtoNCF32(_ x: MTLBuffer, B: Int, C: Int, spatial: Int) -> MTLBuffer {
    let count = B * C * spatial
    let out = MetalContext.shared.bufferPool.get(length: count * 4)
    var c = UInt32(C), s = UInt32(spatial)
    let pipe = KernelCache.shared.pipeline("transpose_sc_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBytes(&c, length: 4, index: 2)
        enc.setBytes(&s, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: C * spatial, height: B, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// General patchify for patchSize > 1 (GPU kernel).
func patchifyGeneral(_ x: Tensor, B: Int, C: Int, F: Int, H: Int, W: Int) -> Tensor {
    let ps = SymmetricPatchifier(patchSize: 1).patchSize  // placeholder
    // TODO: implement for patchSize > 1 if needed
    fatalError("General patchify not implemented — LTX 0.9.8 uses patchSize=1")
}

func unpatchifyGeneral(_ x: Tensor, B: Int, C: Int, F: Int, H: Int, W: Int) -> Tensor {
    fatalError("General unpatchify not implemented — LTX 0.9.8 uses patchSize=1")
}

// MARK: - Element-wise scaled add (for scheduler step)

/// out = scaleA * a + scaleB * b
public func elemAddScaled2(_ a: Tensor, _ b: Tensor, scaleA: Float, scaleB: Float) -> Tensor {
    let out = Tensor.empty(a.shape, dtype: .float16)
    var sA = scaleA, sB = scaleB
    let pipe = KernelCache.shared.pipeline("add_scaled2")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        enc.setBytes(&sA, length: 4, index: 3)
        enc.setBytes(&sB, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: a.count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// out = scaleA * a + scaleB * b (f32)
public func elemAddScaled2F32(_ a: MTLBuffer, _ b: MTLBuffer, scaleA: Float, scaleB: Float, count: Int) -> MTLBuffer {
    let out = MetalContext.shared.bufferPool.get(length: count * 4)
    var sA = scaleA, sB = scaleB
    let pipe = KernelCache.shared.pipeline("add_scaled2_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setBytes(&sA, length: 4, index: 3)
        enc.setBytes(&sB, length: 4, index: 4)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// out = a + scale * b (f32)
public func elemAddScaledF32(_ a: MTLBuffer, _ b: MTLBuffer, scale: Float, count: Int) -> MTLBuffer {
    let out = MetalContext.shared.bufferPool.get(length: count * 4)
    var s = scale
    let pipe = KernelCache.shared.pipeline("add_scaled_f32")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        enc.setBytes(&s, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// out = a + scale * b
public func elemAddScaled(_ a: Tensor, _ b: Tensor, scale: Float) -> Tensor {
    let out = Tensor.empty(a.shape, dtype: .float16)
    var s = scale
    let pipe = KernelCache.shared.pipeline("add_scaled")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(a.buffer, offset: 0, index: 0)
        enc.setBuffer(b.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        enc.setBytes(&s, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: a.count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

// MARK: - DiT Forward Pass

public class DiTModel {
    public let weights: DiTWeights
    public let config: LTXTransformerConfig

    public init(weights: DiTWeights) {
        self.weights = weights
        self.config = weights.config
    }

    /// Forward pass — pure f32 in/out.
    /// hiddenStatesF32: [B, numTokens, inChannels] — patchified latents (f32 MTLBuffer)
    /// cosFreqs/sinFreqs: [numTokens, dim] — precomputed 3D RoPE (f32 MTLBuffer)
    /// encoderHiddenStates: [B, textSeqLen, crossAttentionDim] (f16)
    /// timestep: [B] float — denoising timestep
    /// Returns: [B, numTokens, outChannels] f32 MTLBuffer
    public var forwardLog: ((String) -> Void)? = nil

    public func forward(
        hiddenStatesF32: MTLBuffer,
        cosFreqs: MTLBuffer, sinFreqs: MTLBuffer,
        encoderHiddenStates: Tensor?,
        timestep: Tensor,
        B: Int, numTokens: Int
    ) -> MTLBuffer {
        let dim = config.innerDim
        let pool = MetalContext.shared.bufferPool
        let flog = forwardLog
        let count = B * numTokens * dim

        // 1. Input projection: f32 GEMM (input already f32)
        if let flog {
            let inp = hiddenStatesF32.contents().assumingMemoryBound(to: Float.self)
            let vals = (0..<10).map { inp[$0] }
            flog("input[0:10] f32: \(vals)")
            let wp = weights.patchifyProj.weight.buffer.contents().assumingMemoryBound(to: Float.self)
            let wvals = (0..<10).map { wp[$0] }
            flog("patchifyProj.weight[0:10] f32: \(wvals)")
        }
        var hiddenF32 = weights.patchifyProj.applyF32in(hiddenStatesF32, M: B * numTokens)
        flog?("patchifyProj(f32): \(f32Stats(hiddenF32, count: count))")

        // 2. Timestep embedding — all f32
        let scaledTimestep: Tensor
        if config.timestepScaleMultiplier != 0 {
            scaledTimestep = scaleTimestep(timestep, scale: config.timestepScaleMultiplier, B: B)
        } else {
            scaledTimestep = timestep
        }
        let tsEmb256F32 = timestepEmbeddingF32(scaledTimestep, B: B, dim: 256)
        flog?("tsEmb256(f32): \(f32Stats(tsEmb256F32, count: B * 256))")
        let tsHiddenF32 = weights.adalnTimestepMLP1.applyF32in(tsEmb256F32, M: B)
        flog?("tsMLP1(f32): \(f32Stats(tsHiddenF32, count: B * dim))")
        let tsActF32 = siluF32(tsHiddenF32, count: B * dim)
        let embeddedTimestepF32 = weights.adalnTimestepMLP2.applyF32in(tsActF32, M: B)
        flog?("embeddedTs(f32): \(f32Stats(embeddedTimestepF32, count: B * dim))")
        let siluEmbF32 = siluF32(embeddedTimestepF32, count: B * dim)
        let numAdaGlobal = config.adaptiveNorm == "single_scale" ? 4 : 6
        let timestepEmbF32 = weights.adalnLinear.applyF32in(siluEmbF32, M: B)
        flog?("timestepEmb(f32): \(f32Stats(timestepEmbF32, count: B * numAdaGlobal * dim))")

        // 3. Caption projection — f32 throughout
        var encoderStatesF32: MTLBuffer? = encoderHiddenStates?.buffer
        if let cp1 = weights.captionProj1, let cp2 = weights.captionProj2, let enc = encoderStatesF32 {
            let textSeqLen = encoderHiddenStates!.count / (B * config.captionChannels)
            let projected1 = cp1.applyF32in(enc, M: B * textSeqLen)
            let activated = geluApproximateF32(projected1, count: B * textSeqLen * cp1.N)
            let projected2 = cp2.applyF32in(activated, M: B * textSeqLen)
            encoderStatesF32 = projected2
            flog?("captionProj2(f32): \(f32Stats(projected2, count: B * textSeqLen * cp2.N))")
        }

        // 4. Transformer blocks — all f32, stream weights from mmap
        for i in 0..<weights.blockCount {
            autoreleasepool {
                let block = weights.loadBlock(i)
                MetalContext.shared.beginBatch()
                hiddenF32 = transformerBlock(
                    hiddenF32, block: block,
                    cosFreqs: cosFreqs, sinFreqs: sinFreqs,
                    encoderHiddenStates: encoderStatesF32,
                    encoderCount: encoderStatesF32.map { $0.length / 4 } ?? 0,
                    timestepEmbF32: timestepEmbF32,
                    B: B, numTokens: numTokens,
                    blockIdx: i)
                MetalContext.shared.endBatch()
                flog?("[MEM] block\(i) computed Metal=\(MetalContext.shared.device.currentAllocatedSize/1048576)MB")
                pool.reset(keeping: [hiddenF32, hiddenStatesF32, timestepEmbF32,
                                     embeddedTimestepF32, cosFreqs, sinFreqs]
                           + (encoderStatesF32.map { [$0] } ?? []))
                pool.trimFreeList()
                // block dropped here — autoreleasepool forces immediate Metal buffer dealloc
            }
            let s = pool.stats
            flog?("[MEM] block\(i) freed Metal=\(MetalContext.shared.device.currentAllocatedSize/1048576)MB pool: act=\(s.active/1024)KB/\(s.freeCount)")
        }

        // 5. Output — all f32, return f32 directly
        let normOutF32 = weights.normOut.applyF32(hiddenF32, dim: dim, rows: B * numTokens)
        flog?("normOut(f32): \(f32Stats(normOutF32, count: count))")
        let (finalShiftF32, finalScaleF32) = adaLNCombine2SharedF32(embeddedTimestepF32, table: weights.scaleShiftTable, B: B, C: dim)
        let modulatedF32 = adalnModulateBroadcastF32(normOutF32,
            scale: finalScaleF32, scaleOffset: 0,
            shift: finalShiftF32, shiftOffset: 0,
            numTokens: numTokens, dim: dim, B: B)
        flog?("afterAdaLN(f32): \(f32Stats(modulatedF32, count: count))")
        let projOutF32 = weights.projOut.applyF32in(modulatedF32, M: B * numTokens)
        flog?("projOut(f32): \(f32Stats(projOutF32, count: B * numTokens * config.outChannels))")

        return projOutF32
    }

    /// Transformer block — all f32.
    /// hiddenF32: f32 MTLBuffer [B*numTokens, dim]. Returns f32 MTLBuffer.
    private func transformerBlock(
        _ hiddenF32: MTLBuffer, block: DiTBlockWeights,
        cosFreqs: MTLBuffer, sinFreqs: MTLBuffer,
        encoderHiddenStates: MTLBuffer?,
        encoderCount: Int = 0,
        timestepEmbF32: MTLBuffer,
        B: Int, numTokens: Int,
        blockIdx: Int = -1
    ) -> MTLBuffer {
        let dim = config.innerDim
        let nHeads = config.numAttentionHeads
        let headDim = config.attentionHeadDim
        let count = B * numTokens * dim
        let rows = B * numTokens

        let numAdaParams = block.numAdaParams
        let ada = computeAdaLNParamsF32(timestepEmbF32, table: block.scaleShiftTable,
                                         B: B, dim: dim, numParams: numAdaParams)

        // 1. Self-attention: norm f32→f32, modulate f32, attention f32
        var normHiddenF32: MTLBuffer
        if numAdaParams == 6 {
            normHiddenF32 = block.norm1.applyF32(hiddenF32, dim: dim, rows: rows)
            normHiddenF32 = adalnModulateBroadcastF32(normHiddenF32,
                scale: ada.buffer, scaleOffset: ada.offset(1),
                shift: ada.buffer, shiftOffset: ada.offset(0),
                numTokens: numTokens, dim: dim, B: B)
        } else {
            normHiddenF32 = block.norm1.applyF32(hiddenF32, dim: dim, rows: rows)
            normHiddenF32 = scaleBroadcastF32(normHiddenF32,
                scale: ada.buffer, scaleOffset: ada.offset(0),
                numTokens: numTokens, dim: dim, B: B)
        }

        if blockIdx == 0 { forwardLog?("  normHidden(attn1 input): \(f32Stats(normHiddenF32, count: count))") }
        let attnOutF32 = attentionF32(normHiddenF32, attn: block.attn1,
                                       cosFreqs: cosFreqs, sinFreqs: sinFreqs,
                                       context: nil,
                                       B: B, seqLen: numTokens, nHeads: nHeads, headDim: headDim,
                                       debugBlock: blockIdx == 0)
        if blockIdx == 0 { forwardLog?("  attnOut: \(f32Stats(attnOutF32, count: count))") }

        // Gate f32 * attnOut f32 → f32, add to residual
        let gateIdx = numAdaParams == 6 ? 2 : 1
        let gatedAttn = elemMulBroadcastF32(attnOutF32, gate: ada.buffer, gateOffset: ada.offset(gateIdx),
                                             numTokens: numTokens, dim: dim, B: B)
        if blockIdx == 0 { forwardLog?("  gatedAttn: \(f32Stats(gatedAttn, count: count))") }
        var resF32 = hiddenF32
        elemAddF32(resF32, gatedAttn, count: count)
        if blockIdx == 0 { forwardLog?("  afterSelfAttn: \(f32Stats(resF32, count: count))") }

        // 2. Cross-attention: Q from f32 residual, KV from f32 text
        if let crossAttn = block.attn2, let encStates = encoderHiddenStates {
            let textSeqLen = encoderCount / (B * config.crossAttentionDim)
            let crossOutF32 = attentionF32(resF32, attn: crossAttn,
                                           cosFreqs: cosFreqs, sinFreqs: sinFreqs,
                                           context: encStates,
                                           B: B, seqLen: numTokens, nHeads: nHeads, headDim: headDim,
                                           contextSeqLen: textSeqLen)
            if blockIdx == 0 { forwardLog?("  crossAttnOut: \(f32Stats(crossOutF32, count: count))") }
            elemAddF32(resF32, crossOutF32, count: count)
            if blockIdx == 0 { forwardLog?("  afterCrossAttn: \(f32Stats(resF32, count: count))") }
        }

        // 3. FFN: norm f32→f32, modulate f32, ffn f32
        var normFFF32: MTLBuffer
        if numAdaParams == 6 {
            normFFF32 = block.norm2.applyF32(resF32, dim: dim, rows: rows)
            normFFF32 = adalnModulateBroadcastF32(normFFF32,
                scale: ada.buffer, scaleOffset: ada.offset(4),
                shift: ada.buffer, shiftOffset: ada.offset(3),
                numTokens: numTokens, dim: dim, B: B)
        } else {
            normFFF32 = block.norm2.applyF32(resF32, dim: dim, rows: rows)
            normFFF32 = scaleBroadcastF32(normFFF32,
                scale: ada.buffer, scaleOffset: ada.offset(2),
                numTokens: numTokens, dim: dim, B: B)
        }

        let ffOutF32 = ffnF32(normFFF32, ff: block.ff, M: rows, dim: dim)
        if blockIdx == 0 { forwardLog?("  ffOut: \(f32Stats(ffOutF32, count: count))") }
        let gateMLPIdx = numAdaParams == 6 ? 5 : 3
        let gatedFF = elemMulBroadcastF32(ffOutF32, gate: ada.buffer, gateOffset: ada.offset(gateMLPIdx),
                                           numTokens: numTokens, dim: dim, B: B)
        if blockIdx == 0 { forwardLog?("  gatedFF: \(f32Stats(gatedFF, count: count))") }
        elemAddF32(resF32, gatedFF, count: count)

        return resF32
    }

    /// Multi-head attention — all f32. Returns f32 MTLBuffer.
    /// xF32: f32 buffer for Q (and KV when no context).
    /// context: f16 Tensor for cross-attention KV (cast to f32 inside).
    private func attentionF32(
        _ xF32: MTLBuffer, attn: DiTAttentionWeights,
        cosFreqs: MTLBuffer, sinFreqs: MTLBuffer,
        context: MTLBuffer?,
        B: Int, seqLen: Int, nHeads: Int, headDim: Int,
        contextSeqLen: Int? = nil,
        debugBlock: Bool = false
    ) -> MTLBuffer {
        let dim = nHeads * headDim
        let M = B * seqLen

        // QKV projections — all f32
        var qF32 = attn.toQ.applyF32in(xF32, M: M)

        let kvSeqLen = contextSeqLen ?? seqLen
        let kvM = B * kvSeqLen
        var kF32: MTLBuffer
        var vF32: MTLBuffer
        if let ctx = context {
            // Cross-attention: KV from f32 text embeddings
            kF32 = attn.toK.applyF32in(ctx, M: kvM)
            vF32 = attn.toV.applyF32in(ctx, M: kvM)
        } else {
            kF32 = attn.toK.applyF32in(xF32, M: kvM)
            vF32 = attn.toV.applyF32in(xF32, M: kvM)
        }

        // QK norm (f32→f32)
        if debugBlock { forwardLog?("    Q(pre-norm): \(f32Stats(qF32, count: M * dim))") }
        if debugBlock { forwardLog?("    K(pre-norm): \(f32Stats(kF32, count: kvM * dim))") }
        if let qNorm = attn.qNorm { qF32 = qNorm.applyF32(qF32, dim: dim, rows: M) }
        if let kNorm = attn.kNorm { kF32 = kNorm.applyF32(kF32, dim: dim, rows: kvM) }
        if debugBlock { forwardLog?("    Q(post-norm): \(f32Stats(qF32, count: M * dim))") }
        if debugBlock { forwardLog?("    K(post-norm): \(f32Stats(kF32, count: kvM * dim))") }

        // 3D RoPE f32 (self-attention only) — freqs are [seqLen, dim], kernel broadcasts across batch
        if attn.useRope && !attn.isCrossAttention {
            let qCount = M * dim
            let kCount = kvM * dim
            qF32 = rope3DF32(qF32,
                             cosFreqs: cosFreqs, sinFreqs: sinFreqs,
                             dim: dim, count: qCount, seqLen: seqLen)
            kF32 = rope3DF32(kF32,
                             cosFreqs: cosFreqs, sinFreqs: sinFreqs,
                             dim: dim, count: kCount, seqLen: kvSeqLen)
        }

        if debugBlock { forwardLog?("    Q(post-rope): \(f32Stats(qF32, count: M * dim))") }
        if debugBlock { forwardLog?("    K(post-rope): \(f32Stats(kF32, count: kvM * dim))") }
        if debugBlock { forwardLog?("    V: \(f32Stats(vF32, count: kvM * dim))") }

        // Transpose f32: [seqLen, nHeads, headDim] → [nHeads, seqLen, headDim]
        let qT = transposeSHF32(qF32, seqLen: seqLen, nHeads: nHeads, headDim: headDim)
        let kT = transposeSHF32(kF32, seqLen: kvSeqLen, nHeads: nHeads, headDim: headDim)
        let vT = transposeSHF32(vF32, seqLen: kvSeqLen, nHeads: nHeads, headDim: headDim)

        // Flash attention f32 (bidirectional)
        let attnOutF32 = flashAttentionBidirectionalF32(
            q: qT, k: kT, v: vT,
            R: seqLen, C: kvSeqLen, D: headDim, nHeads: nHeads)

        if debugBlock { forwardLog?("    flashAttnOut: \(f32Stats(attnOutF32, count: nHeads * seqLen * headDim))") }

        // Transpose back f32: [nHeads, seqLen, headDim] → [seqLen, nHeads*headDim]
        let attnFlatF32 = transposeHSF32(attnOutF32, seqLen: seqLen, nHeads: nHeads, headDim: headDim)

        // Output projection f32→f32
        let outF32 = attn.toOut.applyF32in(attnFlatF32, M: M)
        if debugBlock { forwardLog?("    toOut: \(f32Stats(outF32, count: M * dim))") }
        return outF32
    }

    /// FFN hybrid: FA GEMM (f32×f16→f32) for matmuls, f16 for activation.
    /// proj1 in f32, cast to f16 for gelu, cast back to f32, proj2 in f32.
    private func ffnF32(_ x: MTLBuffer, ff: DiTFFNWeights, M: Int, dim: Int) -> MTLBuffer {
        let projF32 = ff.proj.applyF32in(x, M: M)  // [M, projN] f32 — FA GEMM, fast
        let projN = ff.proj.N

        if ff.isGeglu {
            let innerDim = projN / 2
            let halfCount = M * innerDim
            // Cast to f16 for GEGLU activation (halves the memory for the big intermediate)
            let projF16 = castF32toF16(projF32, count: M * projN, shape: [M, projN])
            let out = Tensor.empty([M, innerDim], dtype: .float16)
            let pipe = KernelCache.shared.pipeline("geglu_f16")
            MetalContext.shared.run { enc in
                enc.setComputePipelineState(pipe)
                enc.setBuffer(projF16.buffer, offset: halfCount * 2, index: 0)
                enc.setBuffer(projF16.buffer, offset: 0, index: 1)
                enc.setBuffer(out.buffer, offset: 0, index: 2)
                enc.dispatchThreads(MTLSize(width: halfCount, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
            let outF32 = castF16toF32(out)
            return ff.projOut.applyF32in(outF32, M: M)  // FA GEMM, fast
        } else {
            let count = M * projN
            let activated = geluApproximateF32(projF32, count: count)
            return ff.projOut.applyF32in(activated, M: M)
        }
    }

    /// Fused AdaLN params: single buffer [numParams, B, dim] f32 + offset accessor.
    struct AdaLNParams {
        let buffer: MTLBuffer
        let B: Int
        let dim: Int
        /// Byte offset for param index p.
        func offset(_ p: Int) -> Int { p * B * dim * 4 }
    }

    /// Compute AdaLN params f32: table (f16) + tsEmb (f32) → fused f32 buffer.
    /// Single dispatch writes all params at once (was N separate dispatches).
    private func computeAdaLNParamsF32(_ tsEmbF32: MTLBuffer, table: Tensor,
                                        B: Int, dim: Int, numParams: Int) -> AdaLNParams {
        let totalCount = numParams * B * dim
        let outBuf = MetalContext.shared.bufferPool.get(length: totalCount * 4)
        var dU = UInt32(dim), nU = UInt32(numParams), bU = UInt32(B)
        let pipe = KernelCache.shared.pipeline("ada_ln_params_all_f32")
        MetalContext.shared.run { enc in
            enc.setComputePipelineState(pipe)
            enc.setBuffer(table.buffer, offset: 0, index: 0)
            enc.setBuffer(tsEmbF32, offset: 0, index: 1)
            enc.setBuffer(outBuf, offset: 0, index: 2)
            enc.setBytes(&dU, length: 4, index: 3)
            enc.setBytes(&nU, length: 4, index: 4)
            enc.setBytes(&bU, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: B * dim, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        return AdaLNParams(buffer: outBuf, B: B, dim: dim)
    }
}

/// Broadcast [B, dim] → [B, numTokens, dim] by repeating along seq dimension.
func broadcastToSeq(_ x: Tensor, numTokens: Int, dim: Int) -> Tensor {
    let B = x.count / dim
    if numTokens == 1 { return x }
    let out = Tensor.empty([B, numTokens, dim], dtype: .float16)
    var n = UInt32(numTokens), d = UInt32(dim)
    let pipe = KernelCache.shared.pipeline("broadcast_seq")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(x.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.setBytes(&n, length: 4, index: 2)
        enc.setBytes(&d, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: B * numTokens * dim, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Broadcast RoPE freqs [numTokens, dim] → [B, numTokens, dim] (repeat B times).
/// If B=1, returns as-is.
func broadcastRoPE(_ freqs: Tensor, B: Int, seqLen: Int) -> Tensor {
    if B == 1 { return freqs }
    let dim = freqs.count / seqLen
    let out = Tensor.empty([B, seqLen, dim], dtype: .float16)
    // Simple tile: copy freqs B times
    let bytes = seqLen * dim * 2
    for b in 0..<B {
        memcpy(out.buffer.contents() + b * bytes, freqs.buffer.contents(), bytes)
    }
    return out
}

/// Scale timestep on GPU.
func scaleTimestep(_ t: Tensor, scale: Float, B: Int) -> Tensor {
    // t is [B] float32, multiply by scale
    let out = Tensor.empty([B], dtype: .float32)
    let ptr = t.buffer.contents().assumingMemoryBound(to: Float.self)
    let outPtr = out.buffer.contents().assumingMemoryBound(to: Float.self)
    for i in 0..<B { outPtr[i] = ptr[i] * scale }
    return out
}

/// Compute 1 + scale element-wise.
func onePlusScale(_ scale: Tensor) -> Tensor {
    let out = Tensor.empty(scale.shape, dtype: .float16)
    let pipe = KernelCache.shared.pipeline("one_plus")
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(scale.buffer, offset: 0, index: 0)
        enc.setBuffer(out.buffer, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: scale.count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

// MARK: - Pipeline

public class LTXPipeline {
    public let model: DiTModel
    public let scheduler: RectifiedFlowScheduler
    public let patchifier: SymmetricPatchifier
    public let config: LTXTransformerConfig

    public init(model: DiTModel, scheduler: RectifiedFlowScheduler = RectifiedFlowScheduler(),
                patchifier: SymmetricPatchifier? = nil) {
        self.model = model
        self.scheduler = scheduler
        self.config = model.config
        self.patchifier = patchifier ?? SymmetricPatchifier(patchSize: config.patchSize)
    }

    /// Generate video latents from noise, returning f32 MTLBuffer.
    /// Shape: [1, outChannels, numFrames, height, width] f32
    public func generateLatents(
        textEmbeddings: Tensor,
        textSeqLen: Int,
        numFrames: Int = 2,
        height: Int = 32,
        width: Int = 32,
        numSteps: Int = 20,
        explicitTimesteps: [Float]? = nil,
        guidanceScale: Float = 1.0,
        frameRate: Float = 25.0,
        seed: UInt64 = 42,
        progressHandler: ((Int, Int) -> Void)? = nil,
        log: ((String) -> Void)? = nil
    ) -> MTLBuffer {
        let B = 1
        let latentChannels = config.inChannels
        let numTokens = numFrames * (height / patchifier.patchSize) * (width / patchifier.patchSize)

        if let explicit = explicitTimesteps {
            scheduler.timesteps = explicit
            scheduler.numInferenceSteps = explicit.count
        } else {
            scheduler.setTimesteps(numSteps: numSteps, numTokens: numTokens)
        }
        log?("timesteps: \(scheduler.timesteps.prefix(5))...")

        let totalLatentCount = B * latentChannels * numFrames * height * width
        let noiseF32 = randomNormalF32(count: totalLatentCount, seed: seed)
        // log?("noise f32: \(f32Stats(noiseF32, count: totalLatentCount))")

        var (tokensF32, latentCoords) = patchifier.patchifyF32(noiseF32, B: B, C: latentChannels,
                                                                F: numFrames, H: height, W: width)
        let tokenCount = B * numTokens * latentChannels
        // log?("patchified f32: \(f32Stats(tokensF32, count: tokenCount))")

        let pixelCoords = latentToPixelCoords(latentCoords, frameRate: frameRate)
        let (cosFreqs, sinFreqs) = precomputeFreqsCIS3D(
            indicesGrid: pixelCoords,
            maxPos: config.positionalEmbeddingMaxPos,
            theta: config.positionalEmbeddingTheta,
            dim: config.innerDim)
        log?("rope freqs precomputed (f32)")
        log?("textEmb: shape=\(textEmbeddings.shape)")
        log?("[MEM] pre-loop Metal=\(MetalContext.shared.device.currentAllocatedSize/1048576)MB")

        let pool = MetalContext.shared.bufferPool

        for i in 0..<scheduler.timesteps.count {
            let t = scheduler.timesteps[i]
            let tsTensor = Tensor([t], shape: [B])

            let stepStart = CFAbsoluteTimeGetCurrent()
            let noisePredF32 = model.forward(
                hiddenStatesF32: tokensF32,
                cosFreqs: cosFreqs, sinFreqs: sinFreqs,
                encoderHiddenStates: textEmbeddings,
                timestep: tsTensor,
                B: B, numTokens: numTokens)

            let fwdTime = CFAbsoluteTimeGetCurrent() - stepStart
            tokensF32 = scheduler.stepF32(modelOutput: noisePredF32, timestepIndex: i, sample: tokensF32, count: tokenCount)
            let totalTime = CFAbsoluteTimeGetCurrent() - stepStart
            pool.reset(keeping: [tokensF32, cosFreqs, sinFreqs, textEmbeddings.buffer])
            pool.trimFreeList()
            log?("[PERF] step\(i) t=\(String(format:"%.4f",t)): fwd=\(String(format:"%.1f",fwdTime*1000))ms total=\(String(format:"%.1f",totalTime*1000))ms")

            if i == 0 { model.forwardLog = nil }
            progressHandler?(i + 1, scheduler.timesteps.count)
        }

        // log?("final tokens f32: \(f32Stats(tokensF32, count: tokenCount))")
        let resultF32 = patchifier.unpatchifyF32(tokensF32, B: B, C: config.outChannels,
                                                  F: numFrames, H: height, W: width)
        let resultCount = B * config.outChannels * numFrames * height * width
        // log?("unpatchified f32: \(f32Stats(resultF32, count: resultCount))")
        return resultF32
    }

    /// Denoise existing latents (f32 in, f32 out).
    /// inputLatentsF32: f32 MTLBuffer with shape [B, C, numFrames, height, width]
    public func denoiseLatents(
        inputLatentsF32: MTLBuffer,
        B: Int, C: Int, numFrames: Int, height: Int, width: Int,
        textEmbeddings: Tensor,
        textSeqLen: Int,
        timesteps explicitTimesteps: [Float],
        guidanceScale: Float = 1.0,
        frameRate: Float = 25.0,
        seed: UInt64 = 42,
        noiseOverrideF32: MTLBuffer? = nil,
        progressHandler: ((Int, Int) -> Void)? = nil,
        log: ((String) -> Void)? = nil
    ) -> MTLBuffer {
        let totalCount = B * C * numFrames * height * width

        scheduler.timesteps = explicitTimesteps
        scheduler.numInferenceSteps = explicitTimesteps.count

        // Noise the input latents to first timestep level (all f32)
        let firstT = explicitTimesteps[0]
        let noiseF32 = noiseOverrideF32 ?? randomNormalF32(count: totalCount, seed: seed &+ 1)
        let noisedF32 = elemAddScaled2F32(noiseF32, inputLatentsF32, scaleA: firstT, scaleB: 1.0 - firstT, count: totalCount)
        // log?("noised latents f32 (t=\(firstT)): \(f32Stats(noisedF32, count: totalCount))")

        let numTokens = numFrames * (height / patchifier.patchSize) * (width / patchifier.patchSize)
        var (tokensF32, latentCoords) = patchifier.patchifyF32(noisedF32, B: B, C: C,
                                                                F: numFrames, H: height, W: width)
        let tokenCount = B * numTokens * C

        let pixelCoords = latentToPixelCoords(latentCoords, frameRate: frameRate)
        let (cosFreqs, sinFreqs) = precomputeFreqsCIS3D(
            indicesGrid: pixelCoords,
            maxPos: config.positionalEmbeddingMaxPos,
            theta: config.positionalEmbeddingTheta,
            dim: config.innerDim)

        let pool = MetalContext.shared.bufferPool

        for i in 0..<explicitTimesteps.count {
            let t = explicitTimesteps[i]
            let tsTensor = Tensor([t], shape: [B])

            let stepStart = CFAbsoluteTimeGetCurrent()
            let noisePredF32 = model.forward(
                hiddenStatesF32: tokensF32,
                cosFreqs: cosFreqs, sinFreqs: sinFreqs,
                encoderHiddenStates: textEmbeddings,
                timestep: tsTensor,
                B: B, numTokens: numTokens)

            let fwdTime = CFAbsoluteTimeGetCurrent() - stepStart
            tokensF32 = scheduler.stepF32(modelOutput: noisePredF32, timestepIndex: i, sample: tokensF32, count: tokenCount)
            let totalTime = CFAbsoluteTimeGetCurrent() - stepStart
            pool.reset(keeping: [tokensF32, cosFreqs, sinFreqs, textEmbeddings.buffer])
            pool.trimFreeList()
            log?("[PERF] step\(i) t=\(String(format:"%.4f",t)): fwd=\(String(format:"%.1f",fwdTime*1000))ms total=\(String(format:"%.1f",totalTime*1000))ms")

            progressHandler?(i + 1, explicitTimesteps.count)
        }

        let resultF32 = patchifier.unpatchifyF32(tokensF32, B: B, C: config.outChannels,
                                                  F: numFrames, H: height, W: width)
        let resultCount = B * config.outChannels * numFrames * height * width
        // log?("denoiseF32 result: \(f32Stats(resultF32, count: resultCount))")
        return resultF32
    }
}

// MARK: - Utility

/// Generate random normal f32 buffer (CPU init, then upload).
func randomNormalF32(count: Int, seed: UInt64) -> MTLBuffer {
    srand48(Int(seed))
    var data = [Float](repeating: 0, count: count)
    for i in stride(from: 0, to: count - 1, by: 2) {
        let u1 = max(Float(drand48()), 1e-7)
        let u2 = Float(drand48())
        let r = sqrtf(-2.0 * logf(u1))
        let theta = 2.0 * .pi * u2
        data[i] = r * cosf(theta)
        data[i + 1] = r * sinf(theta)
    }
    if count % 2 != 0 {
        let u1 = max(Float(drand48()), 1e-7)
        let u2 = Float(drand48())
        data[count - 1] = sqrtf(-2.0 * logf(u1)) * cosf(2.0 * .pi * u2)
    }
    return data.withUnsafeBytes {
        MetalContext.shared.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
    }
}

/// Generate random normal f16 tensor (CPU init, then upload).
func randomNormal(shape: [Int], seed: UInt64) -> Tensor {
    srand48(Int(seed))
    let count = shape.reduce(1, *)
    var data = [Float16](repeating: 0, count: count)
    // Box-Muller transform
    for i in stride(from: 0, to: count - 1, by: 2) {
        let u1 = max(Float(drand48()), 1e-7)
        let u2 = Float(drand48())
        let r = sqrtf(-2.0 * logf(u1))
        let theta = 2.0 * .pi * u2
        data[i] = Float16(r * cosf(theta))
        data[i + 1] = Float16(r * sinf(theta))
    }
    if count % 2 != 0 {
        let u1 = max(Float(drand48()), 1e-7)
        let u2 = Float(drand48())
        data[count - 1] = Float16(sqrtf(-2.0 * logf(u1)) * cosf(2.0 * .pi * u2))
    }
    let buf = data.withUnsafeBytes {
        MetalContext.shared.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
    }
    return Tensor(buffer: buf, shape: shape, dtype: .float16)
}

/// Mix decode noise into latents before VAE decode.
/// out = latents * (1 - noiseScale) + noise * noiseScale
public func mixDecodeNoise(_ latents: Tensor, noiseScale: Float, seed: UInt64) -> Tensor {
    if noiseScale <= 0 { return latents }
    let noise = randomNormal(shape: latents.shape, seed: seed)
    let out = Tensor.empty(latents.shape, dtype: .float16)
    let pipe = KernelCache.shared.pipeline("mix_noise")
    var scale = noiseScale
    MetalContext.shared.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(latents.buffer, offset: 0, index: 0)
        enc.setBuffer(noise.buffer, offset: 0, index: 1)
        enc.setBuffer(out.buffer, offset: 0, index: 2)
        enc.setBytes(&scale, length: 4, index: 3)
        enc.dispatchThreads(MTLSize(width: latents.count, height: 1, depth: 1),
                           threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    return out
}

/// Convert latent coords to pixel coords with frame rate scaling.
func latentToPixelCoords(_ coords: [[Float]], frameRate: Float = 25.0,
                          spatialFactor: Int = 32, temporalFactor: Int = 8) -> [[Float]] {
    return [
        coords[0].map { $0 * Float(temporalFactor) / frameRate },
        coords[1].map { $0 * Float(spatialFactor) },
        coords[2].map { $0 * Float(spatialFactor) }
    ]
}

// MARK: - Weight Loading

/// Load DiT weights from safetensors files (diffusers format).
/// All weights stored as f16 on GPU.
public func loadDiTWeights(from url: URL, config: LTXTransformerConfig = LTXTransformerConfig()) throws -> DiTWeights {
    let files = [try SafeTensorsFile(url: url)]
    let dim = config.innerDim

    // Key remapping: diffusers → our names
    func remap(_ key: String) -> String {
        var k = key
        if k.hasPrefix("model.diffusion_model.") { k = String(k.dropFirst("model.diffusion_model.".count)) }
        k = k.replacingOccurrences(of: "proj_in", with: "patchify_proj")
        k = k.replacingOccurrences(of: "time_embed", with: "adaln_single")
        k = k.replacingOccurrences(of: "q_norm", with: "norm_q")
        k = k.replacingOccurrences(of: "k_norm", with: "norm_k")
        return k
    }

    // Load a tensor by name, searching all files. Returns f32 Metal buffer.
    func load(_ name: String) -> Tensor? {
        let remapped = remap(name)
        let prefixed = "model.diffusion_model." + name
        for file in files {
            // Try original, remapped, and prefixed variants
            for tryName in [name, remapped, prefixed] {
                if let info = file.tensors[tryName], let ptr = file.pointer(for: tryName) {
                    let ctx = MetalContext.shared
                    let count = info.shape.reduce(1, *)
                    switch info.dtype {
                    case .float16:
                        // f16 → f32
                        let src = ptr.assumingMemoryBound(to: UInt16.self)
                        let buf = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared)!
                        let dst = buf.contents().assumingMemoryBound(to: Float.self)
                        for i in 0..<count { dst[i] = Float(Float16(bitPattern: src[i])) }
                        return Tensor(buffer: buf, shape: info.shape, dtype: .float32)
                    case .float32:
                        let buf = ctx.device.makeBuffer(bytes: ptr, length: info.byteCount, options: .storageModeShared)!
                        return Tensor(buffer: buf, shape: info.shape, dtype: .float32)
                    case .bfloat16:
                        // bf16 → f32 directly (no f16 intermediate)
                        let src = ptr.assumingMemoryBound(to: UInt16.self)
                        let buf = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared)!
                        let dst = buf.contents().assumingMemoryBound(to: Float.self)
                        let nThreads = min(8, ProcessInfo.processInfo.activeProcessorCount)
                        let chunk = (count + nThreads - 1) / nThreads
                        DispatchQueue.concurrentPerform(iterations: nThreads) { t in
                            let start = t * chunk
                            let end = min(start + chunk, count)
                            for i in start..<end {
                                dst[i] = Float(bitPattern: UInt32(src[i]) << 16)
                            }
                        }
                        return Tensor(buffer: buf, shape: info.shape, dtype: .float32)
                    default:
                        return nil
                    }
                }
            }
        }
        return nil
    }

    func loadRequired(_ name: String) throws -> Tensor {
        guard let t = load(name) else {
            throw DiTLoadError.missingWeight(name)
        }
        return t
    }

    /// Load a tensor and convert to f16 (for bias vectors where f16 is fine).
    func loadAsF16(_ name: String) -> Tensor? {
        guard let t = load(name) else { return nil }
        if t.dtype == .float16 { return t }
        // f32 → f16
        let count = t.count
        let src = t.buffer.contents().assumingMemoryBound(to: Float.self)
        let buf = MetalContext.shared.device.makeBuffer(length: count * 2, options: .storageModeShared)!
        let dst = buf.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<count { dst[i] = Float16(src[i]) }
        return Tensor(buffer: buf, shape: t.shape, dtype: .float16)
    }

    /// Load a tensor in its native format (bf16 stays bf16, f16 stays f16, f32 stays f32).
    /// Zero conversion — just copy raw bytes into an MTLBuffer.
    func loadNative(_ name: String) -> Tensor? {
        let remapped = remap(name)
        let prefixed = "model.diffusion_model." + name
        for file in files {
            for tryName in [name, remapped, prefixed] {
                if let info = file.tensors[tryName], let ptr = file.pointer(for: tryName) {
                    let buf = MetalContext.shared.device.makeBuffer(bytes: ptr, length: info.byteCount, options: .storageModeShared)!
                    let dtype: TensorDType
                    switch info.dtype {
                    case .bfloat16: dtype = .bfloat16
                    case .float16:  dtype = .float16
                    case .float32:  dtype = .float32
                    case .int32:    dtype = .float32  // NF4 packed int32, store as-is
                    default: return nil
                    }
                    return Tensor(buffer: buf, shape: info.shape, dtype: dtype)
                }
            }
        }
        return nil
    }

    func loadLinear(prefix: String, K: Int, N: Int, bias: Bool = true) throws -> DiTLinear {
        // Check for NF4 quantized format: .weight is int32
        let wName = "\(prefix).weight"
        let remappedW = remap(wName)
        var isQ4 = false
        for file in files {
            for tryName in [wName, remappedW] {
                if let info = file.tensors[tryName], info.dtype == .int32 { isQ4 = true; break }
            }
        }
        if isQ4 {
            // NF4: load weight (int32), scales (f16), biases (f16)
            guard let w = loadNative("\(prefix).weight") else {
                throw DiTLoadError.missingWeight("\(prefix).weight")
            }
            let scales = loadNative("\(prefix).scales")
            let biases = loadNative("\(prefix).biases")
            guard let scales, let biases else {
                throw DiTLoadError.missingWeight("\(prefix).scales or .biases")
            }
            let b = bias ? loadAsF16("\(prefix).bias") : nil
            let groupSize = K / scales.shape[1]
            return DiTLinear(weight: w, bias: b, K: K, N: N,
                             q4Scales: scales, q4Biases: biases, q4GroupSize: groupSize)
        }
        // Dense: load weight in native format (bf16 stays bf16 — halves memory)
        guard let w = loadNative("\(prefix).weight") else {
            throw DiTLoadError.missingWeight("\(prefix).weight")
        }
        let b = bias ? loadAsF16("\(prefix).bias") : nil
        return DiTLinear(weight: w, bias: b, K: K, N: N)
    }

    func loadRMSNorm(prefix: String) throws -> DiTRMSNormWeights {
        // RMSNorm weight is typically f32
        let name = "\(prefix).weight"
        for file in files {
            for tryName in [name, remap(name)] {
                if let info = file.tensors[tryName], let ptr = file.pointer(for: tryName) {
                    let count = info.shape.reduce(1, *)
                    let buf: MTLBuffer
                    if info.dtype == .float32 {
                        buf = MetalContext.shared.device.makeBuffer(bytes: ptr, length: count * 4, options: .storageModeShared)!
                    } else if info.dtype == .bfloat16 {
                        // bf16 → f32
                        let src = ptr.assumingMemoryBound(to: UInt16.self)
                        buf = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
                        let dst = buf.contents().assumingMemoryBound(to: Float.self)
                        for i in 0..<count { dst[i] = Float(bitPattern: UInt32(src[i]) << 16) }
                    } else {
                        // f16 → f32
                        let src = ptr.assumingMemoryBound(to: Float16.self)
                        buf = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
                        let dst = buf.contents().assumingMemoryBound(to: Float.self)
                        for i in 0..<count { dst[i] = Float(src[i]) }
                    }
                    return DiTRMSNormWeights(weight: Tensor(buffer: buf, shape: info.shape, dtype: .float32))
                }
            }
        }
        throw DiTLoadError.missingWeight(name)
    }

    func loadLayerNorm(prefix: String) throws -> DiTLayerNormWeights {
        guard let w = loadAsF16("\(prefix).weight"), let b = loadAsF16("\(prefix).bias") else {
            throw DiTLoadError.missingWeight("\(prefix).weight or .bias")
        }
        return DiTLayerNormWeights(weight: w, bias: b)
    }

    // Main model weights
    let patchifyProj = try loadLinear(prefix: "patchify_proj", K: config.inChannels, N: dim)

    // AdaLN timestep embedder
    let tsMLP1 = try loadLinear(prefix: "adaln_single.emb.timestep_embedder.linear_1", K: 256, N: dim)
    let tsMLP2 = try loadLinear(prefix: "adaln_single.emb.timestep_embedder.linear_2", K: dim, N: dim)
    let numAdaGlobal = config.adaptiveNorm == "single_scale" ? 4 : 6
    let adalnLinear = try loadLinear(prefix: "adaln_single.linear", K: dim, N: numAdaGlobal * dim)

    // Caption projection
    let cp1: DiTLinear?
    let cp2: DiTLinear?
    if config.captionChannels > 0 {
        cp1 = try loadLinear(prefix: "caption_projection.linear_1", K: config.captionChannels, N: dim)
        cp2 = try loadLinear(prefix: "caption_projection.linear_2", K: dim, N: dim)
    } else {
        cp1 = nil; cp2 = nil
    }

    // Transformer blocks — lazy loader. Opens a fresh mmap per block, copies weights,
    // then lets the mmap go out of scope so macOS can reclaim pages.
    // Only captures: url, dim, config, remap function (no SafeTensorsFile reference).
    let tensorIndex = files[0].tensors  // just the metadata dictionary (lightweight)

    // Parse header once (no mmap), reuse for all block loads
    let index = try! SafeTensorsIndex(url: url)

    let blockLoader: (Int) -> DiTBlockWeights = { i in
        let prefix = "transformer_blocks.\(i)"

        func resolve(_ name: String) -> String? {
            let remapped = remap(name)
            let prefixed = "model.diffusion_model." + name
            for tryName in [name, remapped, prefixed] {
                if index.tensors[tryName] != nil { return tryName }
            }
            return nil
        }

        // Read tensor directly into Metal buffer via POSIX read (no mmap, no Data)
        func blkLoadNative(_ name: String) -> Tensor? {
            guard let resolved = resolve(name),
                  let info = index.tensors[resolved],
                  let buf = index.readTensorIntoBuffer(resolved) else { return nil }
            let dtype: TensorDType
            switch info.dtype {
            case .bfloat16: dtype = .bfloat16
            case .float16:  dtype = .float16
            case .float32:  dtype = .float32
            case .int32:    dtype = .float32  // NF4 packed int32
            default: return nil
            }
            return Tensor(buffer: buf, shape: info.shape, dtype: dtype)
        }

        func blkLoadAsF16(_ name: String) -> Tensor? {
            guard let resolved = resolve(name),
                  let info = index.tensors[resolved] else { return nil }
            let count = info.shape.reduce(1, *)
            let ctx = MetalContext.shared
            switch info.dtype {
            case .float16:
                // Direct read into final buffer
                guard let buf = index.readTensorIntoBuffer(resolved) else { return nil }
                return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
            case .bfloat16:
                // Read raw into heap, convert into Metal buffer, free heap
                let raw = UnsafeMutablePointer<UInt16>.allocate(capacity: count)
                defer { raw.deallocate() }
                guard index.readTensorRaw(resolved, into: raw) else { return nil }
                let buf = ctx.device.makeBuffer(length: count * 2, options: .storageModeShared)!
                let dst = buf.contents().assumingMemoryBound(to: Float16.self)
                for j in 0..<count { dst[j] = Float16(Float(bitPattern: UInt32(raw[j]) << 16)) }
                return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
            case .float32:
                let raw = UnsafeMutablePointer<Float>.allocate(capacity: count)
                defer { raw.deallocate() }
                guard index.readTensorRaw(resolved, into: raw) else { return nil }
                let buf = ctx.device.makeBuffer(length: count * 2, options: .storageModeShared)!
                let dst = buf.contents().assumingMemoryBound(to: Float16.self)
                for j in 0..<count { dst[j] = Float16(raw[j]) }
                return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
            default:
                return nil
            }
        }

        func blkLoadLinear(_ prefix: String, K: Int, N: Int, bias: Bool = true) -> DiTLinear {
            let w = blkLoadNative("\(prefix).weight")!
            let b = bias ? blkLoadAsF16("\(prefix).bias") : nil
            // Check for NF4: if scales tensor exists
            if let scales = blkLoadNative("\(prefix).scales"),
               let biases = blkLoadNative("\(prefix).biases") {
                let groupSize = K / scales.shape[1]
                return DiTLinear(weight: w, bias: b, K: K, N: N,
                                 q4Scales: scales, q4Biases: biases, q4GroupSize: groupSize)
            }
            return DiTLinear(weight: w, bias: b, K: K, N: N)
        }

        func blkLoadRMSNorm(_ prefix: String) -> DiTRMSNormWeights {
            guard let resolved = resolve("\(prefix).weight"),
                  let info = index.tensors[resolved] else {
                fatalError("Missing RMSNorm weight: \(prefix).weight")
            }
            let count = info.shape.reduce(1, *)
            let ctx = MetalContext.shared
            let buf: MTLBuffer
            if info.dtype == .float32 {
                // Direct read into final buffer
                buf = index.readTensorIntoBuffer(resolved)!
            } else if info.dtype == .bfloat16 {
                let raw = UnsafeMutablePointer<UInt16>.allocate(capacity: count)
                defer { raw.deallocate() }
                _ = index.readTensorRaw(resolved, into: raw)
                buf = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared)!
                let dst = buf.contents().assumingMemoryBound(to: Float.self)
                for j in 0..<count { dst[j] = Float(bitPattern: UInt32(raw[j]) << 16) }
            } else {
                let raw = UnsafeMutablePointer<Float16>.allocate(capacity: count)
                defer { raw.deallocate() }
                _ = index.readTensorRaw(resolved, into: raw)
                buf = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared)!
                let dst = buf.contents().assumingMemoryBound(to: Float.self)
                for j in 0..<count { dst[j] = Float(raw[j]) }
            }
            return DiTRMSNormWeights(weight: Tensor(buffer: buf, shape: info.shape, dtype: .float32))
        }

        // norm1/norm2 are not in the safetensors — they use default weight=1
        let norm1 = DiTRMSNormWeights(weight: Tensor([Float](repeating: 1.0, count: dim), shape: [dim]))

        let selfAttnQ = blkLoadLinear("\(prefix).attn1.to_q", K: dim, N: dim)
        let selfAttnK = blkLoadLinear("\(prefix).attn1.to_k", K: dim, N: dim)
        let selfAttnV = blkLoadLinear("\(prefix).attn1.to_v", K: dim, N: dim)
        let selfAttnOut = blkLoadLinear("\(prefix).attn1.to_out.0", K: dim, N: dim)
        let qNorm: DiTRMSNormWeights?
        let kNorm: DiTRMSNormWeights?
        if config.qkNorm {
            qNorm = blkLoadRMSNorm("\(prefix).attn1.q_norm")
            kNorm = blkLoadRMSNorm("\(prefix).attn1.k_norm")
        } else {
            qNorm = nil; kNorm = nil
        }

        let selfAttn = DiTAttentionWeights(
            toQ: selfAttnQ, toK: selfAttnK, toV: selfAttnV, toOut: selfAttnOut,
            qNorm: qNorm, kNorm: kNorm, useRope: true, isCrossAttention: false)

        let norm2 = DiTRMSNormWeights(weight: Tensor([Float](repeating: 1.0, count: dim), shape: [dim]))

        var crossAttn: DiTAttentionWeights? = nil
        if config.crossAttentionDim > 0 {
            let crossQ = blkLoadLinear("\(prefix).attn2.to_q", K: dim, N: dim)
            let crossK = blkLoadLinear("\(prefix).attn2.to_k", K: config.crossAttentionDim, N: dim)
            let crossV = blkLoadLinear("\(prefix).attn2.to_v", K: config.crossAttentionDim, N: dim)
            let crossOut = blkLoadLinear("\(prefix).attn2.to_out.0", K: dim, N: dim)
            let crossQNorm: DiTRMSNormWeights?
            let crossKNorm: DiTRMSNormWeights?
            if config.qkNorm {
                crossQNorm = blkLoadRMSNorm("\(prefix).attn2.q_norm")
                crossKNorm = blkLoadRMSNorm("\(prefix).attn2.k_norm")
            } else {
                crossQNorm = nil; crossKNorm = nil
            }
            crossAttn = DiTAttentionWeights(
                toQ: crossQ, toK: crossK, toV: crossV, toOut: crossOut,
                qNorm: crossQNorm, kNorm: crossKNorm, useRope: true, isCrossAttention: true)
        }

        let isGeglu = config.activationFn == "geglu"
        let innerDim = dim * 4
        let ffProjN = isGeglu ? innerDim * 2 : innerDim
        let ffProj = blkLoadLinear("\(prefix).ff.net.0.proj", K: dim, N: ffProjN)
        let ffProjOut = blkLoadLinear("\(prefix).ff.net.2", K: innerDim, N: dim)
        let ff = DiTFFNWeights(proj: ffProj, projOut: ffProjOut, isGeglu: isGeglu)

        let numParams = config.adaptiveNorm == "single_scale" ? 4 : 6
        let table = blkLoadAsF16("\(prefix).scale_shift_table")!

        return DiTBlockWeights(
            norm1: norm1, attn1: selfAttn, norm2: norm2, attn2: crossAttn,
            ff: ff, scaleShiftTable: table, numAdaParams: numParams)
        // No mmap — all reads via POSIX read() into Metal buffers
    }

    // Output — norm_out uses default weight=1, bias=0 (not in safetensors)
    let normOutW: Tensor
    let normOutB: Tensor
    if let w = loadAsF16("norm_out.weight"), let b = loadAsF16("norm_out.bias") {
        normOutW = w; normOutB = b
    } else {
        let onesF16 = [Float16](repeating: 1.0, count: dim)
        let zerosF16 = [Float16](repeating: 0.0, count: dim)
        let ctx = MetalContext.shared
        let wBuf = onesF16.withUnsafeBytes { ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        let bBuf = zerosF16.withUnsafeBytes { ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        normOutW = Tensor(buffer: wBuf, shape: [dim], dtype: .float16)
        normOutB = Tensor(buffer: bBuf, shape: [dim], dtype: .float16)
    }
    let normOut = DiTLayerNormWeights(weight: normOutW, bias: normOutB)
    guard let scaleShiftTable = loadAsF16("scale_shift_table") else {
        throw DiTLoadError.missingWeight("scale_shift_table")
    }
    let projOut = try loadLinear(prefix: "proj_out", K: dim, N: config.outChannels)

    return DiTWeights(
        patchifyProj: patchifyProj,
        adalnTimestepMLP1: tsMLP1,
        adalnTimestepMLP2: tsMLP2,
        adalnLinear: adalnLinear,
        captionProj1: cp1,
        captionProj2: cp2,
        blockCount: config.numLayers,
        loadBlock: blockLoader,
        normOut: normOut,
        scaleShiftTable: scaleShiftTable,
        projOut: projOut,
        config: config)
}

/// Load pre-computed T5 text embeddings from safetensors.
/// Returns (embeddings [1, seqLen, 4096], mask [1, seqLen]).
/// Embeddings are kept as f32 for precision; mask is f16.
public func loadTextEmbeddings(from url: URL) throws -> (Tensor, Tensor?) {
    let file = try SafeTensorsFile(url: url)
    let ctx = MetalContext.shared

    func loadAsF32(_ name: String) -> Tensor? {
        return file.withPointer(for: name) { ptr, info in
            if info.dtype == .float32 {
                let buf = ctx.device.makeBuffer(bytes: ptr, length: info.byteCount, options: .storageModeShared)!
                return Tensor(buffer: buf, shape: info.shape, dtype: .float32)
            } else if info.dtype == .float16 {
                let count = info.byteCount / 2
                let src = ptr.assumingMemoryBound(to: UInt16.self)
                let buf = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared)!
                let dst = buf.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<count { dst[i] = Float(Float16(bitPattern: src[i])) }
                return Tensor(buffer: buf, shape: info.shape, dtype: .float32)
            } else if info.dtype == .bfloat16 {
                let count = info.byteCount / 2
                let src = ptr.assumingMemoryBound(to: UInt16.self)
                let buf = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared)!
                let dst = buf.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<count { dst[i] = bfloat16ToFloat32(src[i]) }
                return Tensor(buffer: buf, shape: info.shape, dtype: .float32)
            }
            return nil
        } ?? nil
    }

    func loadF16(_ name: String) -> Tensor? {
        return file.withPointer(for: name) { ptr, info in
            if info.dtype == .float16 {
                let buf = ctx.device.makeBuffer(bytes: ptr, length: info.byteCount, options: .storageModeShared)!
                return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
            }
            return nil
        } ?? nil
    }

    var emb = loadAsF32("prompt_embeds") ?? loadAsF32("hidden_states") ?? loadAsF32("encoder_hidden_states") ?? loadAsF32("text_embeddings")
    guard var embeddings = emb else { throw DiTLoadError.missingWeight("text embeddings") }

    // Add batch dim if 2D: [seqLen, dim] → [1, seqLen, dim]
    if embeddings.shape.count == 2 {
        embeddings = Tensor(buffer: embeddings.buffer, shape: [1, embeddings.shape[0], embeddings.shape[1]], dtype: embeddings.dtype)
    }

    // Mask may be f32 or f16 in the file — try both loaders
    let mask = loadAsF32("attention_mask") ?? loadAsF32("encoder_attention_mask") ?? loadF16("attention_mask") ?? loadF16("encoder_attention_mask")
    return (embeddings, mask)
}

/// Quick min/max/mean/nan stats for an f16 tensor (CPU readback).
public func f16Stats(_ t: Tensor) -> String {
    let count = t.count
    guard count > 0 else { return "empty" }
    let ptr = t.buffer.contents().assumingMemoryBound(to: UInt16.self)
    var mn = Float.infinity, mx = -Float.infinity, sum: Float = 0
    var nanCount = 0, infCount = 0
    let stride = max(1, count / 10000)  // sample for large tensors
    var sampled = 0
    for i in Swift.stride(from: 0, to: count, by: stride) {
        let v = Float(Float16(bitPattern: ptr[i]))
        if v.isNaN { nanCount += 1; continue }
        if v.isInfinite { infCount += 1 }
        if v < mn { mn = v }
        if v > mx { mx = v }
        sum += v
        sampled += 1
    }
    let mean = sampled > 0 ? sum / Float(sampled) : 0
    return String(format: "min=%.4f max=%.4f mean=%.4f nan=%d inf=%d n=%d", mn, mx, mean, nanCount, infCount, count)
}

public func f32Stats(_ buf: MTLBuffer, count: Int) -> String {
    guard count > 0 else { return "empty" }
    let ptr = buf.contents().assumingMemoryBound(to: Float.self)
    var mn = Float.infinity, mx = -Float.infinity, sum: Float = 0
    var nanCount = 0, infCount = 0
    var sampled = 0
    for i in 0..<count {
        let v = ptr[i]
        if v.isNaN { nanCount += 1; continue }
        if v.isInfinite { infCount += 1 }
        if v < mn { mn = v }
        if v > mx { mx = v }
        sum += v
        sampled += 1
    }
    let mean = sampled > 0 ? sum / Float(sampled) : 0
    return String(format: "min=%.4f max=%.4f mean=%.6f nan=%d inf=%d n=%d", mn, mx, mean, nanCount, infCount, count)
}

public enum DiTLoadError: Error {
    case missingWeight(String)
}
