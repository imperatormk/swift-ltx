// Latent Upsampler for LTX-Video multi-scale pipeline.
// CNN that 2x spatially upsamples latents: Conv3d → GN → SiLU → 4×ResBlock → Conv2d+PixelShuffle → 4×ResBlock → Conv3d
// Port of ltxv-spatial-upscaler-0.9.8.
// All f32 — no f16 path.
// Lazy-loaded: weights loaded per-phase from mmap, freed after each phase.

import Metal

// MARK: - Weight containers

public struct Conv3DWeightF32 {
    public let weight: MTLBuffer  // [C_out, C_in, kD, kH, kW] f32
    public let bias: MTLBuffer    // [C_out] f32
    public let C_in: Int, C_out: Int
    public let kD: Int, kH: Int, kW: Int
}

public struct UpsamplerResBlockWeights {
    public let conv1: Conv3DWeightF32
    public let norm1Weight: Tensor  // [C] f32
    public let norm1Bias: Tensor    // [C] f32
    public let conv2: Conv3DWeightF32
    public let norm2Weight: Tensor  // [C] f32
    public let norm2Bias: Tensor    // [C] f32
}

// MARK: - Lazy Weight Loader

/// Holds an open SafeTensorsFile and loads upsampler weights on demand.
public class UpsamplerWeightLoader {
    let file: SafeTensorsFile
    let midChannels: Int = 512

    public init(url: URL) throws {
        self.file = try SafeTensorsFile(url: url)
    }

    func loadBufF32(_ name: String) throws -> MTLBuffer {
        guard let info = file.tensors[name], let ptr = file.pointer(for: name) else {
            throw DiTLoadError.missingWeight(name)
        }
        let count = info.shape.reduce(1, *)
        let ctx = MetalContext.shared
        if info.dtype == .float32 {
            return ctx.device.makeBuffer(bytes: ptr, length: count * 4, options: .storageModeShared)!
        } else if info.dtype == .float16 {
            let src = ptr.assumingMemoryBound(to: Float16.self)
            let buf = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared)!
            let dst = buf.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<count { dst[i] = Float(src[i]) }
            return buf
        } else if info.dtype == .bfloat16 {
            let src = ptr.assumingMemoryBound(to: UInt16.self)
            let buf = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared)!
            let dst = buf.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<count { dst[i] = Float(bitPattern: UInt32(src[i]) << 16) }
            return buf
        }
        throw VideoUpsamplerError.unsupportedDType(name: name, dtype: info.dtype)
    }

    func loadTensorF32(_ name: String) throws -> Tensor {
        guard let info = file.tensors[name], let ptr = file.pointer(for: name) else {
            throw DiTLoadError.missingWeight(name)
        }
        let count = info.shape.reduce(1, *)
        let ctx = MetalContext.shared
        if info.dtype == .float32 {
            let buf = ctx.device.makeBuffer(bytes: ptr, length: count * 4, options: .storageModeShared)!
            return Tensor(buffer: buf, shape: info.shape, dtype: .float32)
        } else if info.dtype == .float16 {
            let src = ptr.assumingMemoryBound(to: Float16.self)
            let buf = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared)!
            let dst = buf.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<count { dst[i] = Float(src[i]) }
            return Tensor(buffer: buf, shape: info.shape, dtype: .float32)
        } else if info.dtype == .bfloat16 {
            let src = ptr.assumingMemoryBound(to: UInt16.self)
            let buf = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared)!
            let dst = buf.contents().assumingMemoryBound(to: Float.self)
            for i in 0..<count { dst[i] = Float(bitPattern: UInt32(src[i]) << 16) }
            return Tensor(buffer: buf, shape: info.shape, dtype: .float32)
        }
        throw VideoUpsamplerError.unsupportedDType(name: name, dtype: info.dtype)
    }

    func loadConv3d(_ prefix: String, C_in: Int, C_out: Int, kD: Int = 3, kH: Int = 3, kW: Int = 3) throws -> Conv3DWeightF32 {
        let w = try loadBufF32("\(prefix).weight")
        let b = try loadBufF32("\(prefix).bias")
        return Conv3DWeightF32(weight: w, bias: b, C_in: C_in, C_out: C_out, kD: kD, kH: kH, kW: kW)
    }

    func loadConv2dAsConv3d(_ prefix: String, C_in: Int, C_out: Int) throws -> Conv3DWeightF32 {
        let w = try loadBufF32("\(prefix).weight")
        let b = try loadBufF32("\(prefix).bias")
        return Conv3DWeightF32(weight: w, bias: b, C_in: C_in, C_out: C_out, kD: 1, kH: 3, kW: 3)
    }

    func loadResBlock(_ prefix: String, C: Int) throws -> UpsamplerResBlockWeights {
        return UpsamplerResBlockWeights(
            conv1: try loadConv3d("\(prefix).conv1", C_in: C, C_out: C),
            norm1Weight: try loadTensorF32("\(prefix).norm1.weight"),
            norm1Bias: try loadTensorF32("\(prefix).norm1.bias"),
            conv2: try loadConv3d("\(prefix).conv2", C_in: C, C_out: C),
            norm2Weight: try loadTensorF32("\(prefix).norm2.weight"),
            norm2Bias: try loadTensorF32("\(prefix).norm2.bias")
        )
    }

    /// Load initial conv + norm weights.
    func loadInitial() throws -> (conv: Conv3DWeightF32, normW: Tensor, normB: Tensor) {
        let conv = try loadConv3d("initial_conv", C_in: 128, C_out: midChannels)
        let nw = try loadTensorF32("initial_norm.weight")
        let nb = try loadTensorF32("initial_norm.bias")
        return (conv, nw, nb)
    }

    /// Load a single resblock.
    func loadPreResBlock(_ i: Int) throws -> UpsamplerResBlockWeights {
        try loadResBlock("res_blocks.\(i)", C: midChannels)
    }

    /// Load upsample conv (512→2048).
    func loadUpsampleConv() throws -> Conv3DWeightF32 {
        try loadConv2dAsConv3d("upsampler.0", C_in: midChannels, C_out: midChannels * 4)
    }

    /// Load a single post-resblock.
    func loadPostResBlock(_ i: Int) throws -> UpsamplerResBlockWeights {
        try loadResBlock("post_upsample_res_blocks.\(i)", C: midChannels)
    }

    /// Load final conv (512→128).
    func loadFinalConv() throws -> Conv3DWeightF32 {
        try loadConv3d("final_conv", C_in: midChannels, C_out: 128)
    }
}

// MARK: - Upsampler Forward (lazy-loaded)

/// Forward f32: [B, 128, F, H, W] f32 → [B, 128, F, H*2, W*2] f32.
/// Weights loaded per-phase from mmap, freed after each phase.
public func upsamplerForward(_ latent: MTLBuffer, loader: UpsamplerWeightLoader,
                              B: Int, F: Int, H: Int, W: Int,
                              log: ((String) -> Void)? = nil) throws -> MTLBuffer {
    let C = loader.midChannels  // 512
    let pool = MetalContext.shared.bufferPool
    var t0 = CFAbsoluteTimeGetCurrent()
    func perf(_ label: String) {
        let now = CFAbsoluteTimeGetCurrent()
        log?("[PERF] \(label): \(String(format: "%.1fms", (now - t0) * 1000))")
        t0 = now
    }

    // Phase 1: Initial conv3d + groupnorm + silu
    let initial = try loader.loadInitial()
    var x: MTLBuffer = autoreleasepool {
        var h = conv3dF32(latent, weight: initial.conv.weight, bias: initial.conv.bias,
                          B: B, C_in: 128, D: F, H: H, W: W,
                          C_out: C, kD: 3, kH: 3, kW: 3, pD: 1, pH: 1, pW: 1)
        perf("initial_conv3d(128→512)")
        h = groupNorm3dF32(h, weight: initial.normW, bias: initial.normB,
                           B: B, C: C, spatialSize: F * H * W, numGroups: 32)
        perf("initial_groupnorm")
        h = siluF32(h, count: B * C * F * H * W)
        perf("initial_silu")
        pool.reset(keeping: [h])
        return h
    }

    // Phase 2: Pre-upsample ResBlocks (loaded one at a time)
    for i in 0..<4 {
        let rb = try loader.loadPreResBlock(i)
        x = autoreleasepool {
            let out = resBlockF32(x, weights: rb, B: B, C: C, D: F, H: H, W: W, idx: "pre\(i)", log: log)
            pool.reset(keeping: [out])
            return out
        }
    }

    // Phase 3: Spatial upsample (512→2048 conv + pixel shuffle)
    let newH = H * 2, newW = W * 2
    let upConv = try loader.loadUpsampleConv()
    x = autoreleasepool {
        t0 = CFAbsoluteTimeGetCurrent()
        let folded = transposeSHF32(x, seqLen: C, nHeads: F, headDim: H * W)
        pool.reset(keeping: [folded]); perf("transpose_CF→FC")
        let up2048 = conv3dF32(folded, weight: upConv.weight, bias: upConv.bias,
                               B: B * F, C_in: C, D: 1, H: H, W: W,
                               C_out: C * 4, kD: 1, kH: 3, kW: 3, pD: 0, pH: 1, pW: 1)
        pool.reset(keeping: [up2048]); perf("upsample_conv2d(512→2048)")
        let shuffled = pixelShuffle3dF32(up2048, B: B * F, C_out: C, D_in: 1, H_in: H, W_in: W, p1: 1, p2: 2, p3: 2)
        pool.reset(keeping: [shuffled]); perf("pixel_shuffle_2x")
        let out = transposeHSF32(shuffled, seqLen: C, nHeads: F, headDim: newH * newW)
        pool.reset(keeping: [out]); perf("transpose_FC→CF")
        return out
    }

    // Phase 4: Post-upsample ResBlocks (loaded one at a time)
    for i in 0..<4 {
        let rb = try loader.loadPostResBlock(i)
        x = autoreleasepool {
            let out = resBlockF32(x, weights: rb, B: B, C: C, D: F, H: newH, W: newW, idx: "post\(i)", log: log)
            pool.reset(keeping: [out])
            return out
        }
    }

    // Phase 5: Final conv3d: 512 → 128
    let finalConv = try loader.loadFinalConv()
    x = autoreleasepool {
        t0 = CFAbsoluteTimeGetCurrent()
        let out = conv3dF32(x, weight: finalConv.weight, bias: finalConv.bias,
                            B: B, C_in: C, D: F, H: newH, W: newW,
                            C_out: 128, kD: 3, kH: 3, kW: 3, pD: 1, pH: 1, pW: 1)
        perf("final_conv3d(512→128)")
        pool.reset(keeping: [out])
        return out
    }

    return x
}

public enum VideoUpsamplerError: Error, CustomStringConvertible {
    case unsupportedDType(name: String, dtype: TensorDType)

    public var description: String {
        switch self {
        case .unsupportedDType(let name, let dtype):
            return "Unsupported upsampler dtype for \(name): \(dtype.rawValue)"
        }
    }
}

/// ResBlock helper for upsampler (f32).
/// Pool-resets aggressively between ops so peak = 2× activation_size (x + current h).
private func resBlockF32(_ x: MTLBuffer, weights w: UpsamplerResBlockWeights,
                          B: Int, C: Int, D: Int, H: Int, W: Int,
                          idx: String = "", log: ((String) -> Void)? = nil) -> MTLBuffer {
    let spatial = D * H * W
    let count = B * C * spatial
    let pool = MetalContext.shared.bufferPool
    var t0 = CFAbsoluteTimeGetCurrent()
    func perf(_ label: String) {
        let now = CFAbsoluteTimeGetCurrent()
        log?("[PERF] rb_\(idx).\(label): \(String(format: "%.1fms", (now - t0) * 1000))")
        t0 = now
    }
    var h = conv3dF32(x, weight: w.conv1.weight, bias: w.conv1.bias,
                      B: B, C_in: C, D: D, H: H, W: W,
                      C_out: C, kD: 3, kH: 3, kW: 3, pD: 1, pH: 1, pW: 1)
    pool.reset(keeping: [x, h]); perf("conv1")
    h = groupNorm3dF32(h, weight: w.norm1Weight, bias: w.norm1Bias,
                       B: B, C: C, spatialSize: spatial, numGroups: 32)
    pool.reset(keeping: [x, h]); perf("gn1")
    h = siluF32(h, count: count)
    pool.reset(keeping: [x, h]); perf("silu1")
    h = conv3dF32(h, weight: w.conv2.weight, bias: w.conv2.bias,
                  B: B, C_in: C, D: D, H: H, W: W,
                  C_out: C, kD: 3, kH: 3, kW: 3, pD: 1, pH: 1, pW: 1)
    pool.reset(keeping: [x, h]); perf("conv2")
    h = groupNorm3dF32(h, weight: w.norm2Weight, bias: w.norm2Bias,
                       B: B, C: C, spatialSize: spatial, numGroups: 32)
    pool.reset(keeping: [x, h]); perf("gn2")
    // In-place add x into h (frees x's slot), then silu in-place
    elemAddF32(h, x, count: count)
    pool.reset(keeping: [h]); perf("add")
    let out = siluF32(h, count: count)
    pool.reset(keeping: [out]); perf("silu2")
    return out
}

// MARK: - AdaIN

/// AdaIN f32: match per-channel mean/std of latents to reference. All f32 MTLBuffers.
/// GPU-accelerated: parallel reduction for per-channel statistics, then CPU computes scale/shift,
/// then GPU applies channel_scale_bias.
public func adainFilter(_ latents: MTLBuffer, reference: MTLBuffer,
                         B: Int, C: Int, latShape: [Int], refShape: [Int],
                         factor: Float = 1.0) -> MTLBuffer {
    let latSpatial = latShape[2] * latShape[3] * latShape[4]
    let refSpatial = refShape[2] * refShape[3] * refShape[4]
    let bc = B * C
    let ctx = MetalContext.shared

    // GPU: compute per-channel mean and variance for both latents and reference
    let latMeanBuf = ctx.bufferPool.get(length: bc * 4)
    let latVarBuf = ctx.bufferPool.get(length: bc * 4)
    let refMeanBuf = ctx.bufferPool.get(length: bc * 4)
    let refVarBuf = ctx.bufferPool.get(length: bc * 4)

    let tgSize = 256
    let pipe = KernelCache.shared.pipeline("channel_mean_var_f32")
    var cU = UInt32(C)
    var latSpatU = UInt32(latSpatial)
    var refSpatU = UInt32(refSpatial)

    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(latents, offset: 0, index: 0)
        enc.setBuffer(latMeanBuf, offset: 0, index: 1)
        enc.setBuffer(latVarBuf, offset: 0, index: 2)
        enc.setBytes(&cU, length: 4, index: 3)
        enc.setBytes(&latSpatU, length: 4, index: 4)
        enc.setThreadgroupMemoryLength(tgSize * 2 * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: bc, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }
    ctx.run { enc in
        enc.setComputePipelineState(pipe)
        enc.setBuffer(reference, offset: 0, index: 0)
        enc.setBuffer(refMeanBuf, offset: 0, index: 1)
        enc.setBuffer(refVarBuf, offset: 0, index: 2)
        enc.setBytes(&cU, length: 4, index: 3)
        enc.setBytes(&refSpatU, length: 4, index: 4)
        enc.setThreadgroupMemoryLength(tgSize * 2 * 4, index: 0)
        enc.dispatchThreadgroups(MTLSize(width: bc, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
    }

    // If batching, flush so GPU results are available on CPU
    if ctx.batching {
        ctx.endBatch()
        ctx.beginBatch()
    }

    // CPU: compute per-channel scale and shift from statistics
    let latMean = latMeanBuf.contents().assumingMemoryBound(to: Float.self)
    let latVar = latVarBuf.contents().assumingMemoryBound(to: Float.self)
    let refMean = refMeanBuf.contents().assumingMemoryBound(to: Float.self)
    let refVar = refVarBuf.contents().assumingMemoryBound(to: Float.self)

    var scaleData = [Float](repeating: 0, count: bc)
    var shiftData = [Float](repeating: 0, count: bc)
    for i in 0..<bc {
        let refStd = sqrtf(max(refVar[i], 1e-8))
        let latStd = sqrtf(max(latVar[i], 1e-8))
        let sc = refStd / latStd
        let sh = refMean[i] - sc * latMean[i]
        scaleData[i] = 1.0 - factor + factor * sc
        shiftData[i] = factor * sh
    }

    let scaleBuf = scaleData.withUnsafeBytes {
        ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
    }
    let shiftBuf = shiftData.withUnsafeBytes {
        ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
    }
    return channelScaleBiasF32(x: latents, scale: scaleBuf, bias: shiftBuf,
                                B: B, C: C, spatial: latSpatial)
}
