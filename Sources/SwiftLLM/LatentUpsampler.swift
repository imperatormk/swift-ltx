// Latent Upsampler for LTX-Video multi-scale pipeline.
// CNN that 2x spatially upsamples latents: Conv3d → GN → SiLU → 4×ResBlock → Conv2d+PixelShuffle → 4×ResBlock → Conv3d
// Port of ltxv-spatial-upscaler-0.9.8.
// All f32 — no f16 path.

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

public struct UpsamplerWeights {
    public let initialConv: Conv3DWeightF32
    public let initialNormWeight: Tensor       // [512] f32
    public let initialNormBias: Tensor         // [512] f32
    public let resBlocks: [UpsamplerResBlockWeights]
    public let upsampleConv: Conv3DWeightF32
    public let postResBlocks: [UpsamplerResBlockWeights]
    public let finalConv: Conv3DWeightF32
    public let midChannels: Int
}

// MARK: - Upsampler Forward

public class LatentUpsampler {
    public let weights: UpsamplerWeights

    public init(weights: UpsamplerWeights) {
        self.weights = weights
    }

    /// Forward f32: [B, 128, F, H, W] f32 MTLBuffer → [B, 128, F, H*2, W*2] f32 MTLBuffer
    public func forward(_ latent: MTLBuffer, B: Int, F: Int, H: Int, W: Int,
                        log: ((String) -> Void)? = nil) -> MTLBuffer {
        let C = weights.midChannels  // 512
        let pool = MetalContext.shared.bufferPool

        // Initial conv3d + groupnorm + silu
        var x = conv3dF32(latent, weight: weights.initialConv.weight, bias: weights.initialConv.bias,
                          B: B, C_in: 128, D: F, H: H, W: W,
                          C_out: C, kD: 3, kH: 3, kW: 3, pD: 1, pH: 1, pW: 1)
        log?("initial_conv: \(f32Stats(x, count: B*C*F*H*W))")
        x = groupNorm3dF32(x, weight: weights.initialNormWeight, bias: weights.initialNormBias,
                           B: B, C: C, spatialSize: F * H * W, numGroups: 32)
        x = siluF32(x, count: B * C * F * H * W)
        pool.reset(keeping: [x])
        log?("after initial: \(f32Stats(x, count: B*C*F*H*W))")

        // Pre-upsample ResBlocks
        for (i, rb) in weights.resBlocks.enumerated() {
            x = resBlock(x, weights: rb, B: B, C: C, D: F, H: H, W: W)
            pool.reset(keeping: [x])
            log?("resBlock\(i): \(f32Stats(x, count: B*C*F*H*W))")
        }

        // Spatial upsample: fold frames → Conv2d (as Conv3d D=1) → PixelShuffle2D → unfold
        // x is [B, C, F, H, W] (NCFHW). Conv3d expects [B*F, C, 1, H, W] contiguous,
        // so we must transpose [C, F, HW] → [F, C, HW] first.
        let folded = transposeSHF32(x, seqLen: C, nHeads: F, headDim: H * W)
        // folded is now [B, F, C, H, W] = [B*F, C, H, W] contiguous
        var up = conv3dF32(folded, weight: weights.upsampleConv.weight, bias: weights.upsampleConv.bias,
                           B: B * F, C_in: C, D: 1, H: H, W: W,
                           C_out: C * 4, kD: 1, kH: 3, kW: 3, pD: 0, pH: 1, pW: 1)
        log?("upsample_conv: \(f32Stats(up, count: B*F*C*4*H*W))")
        up = pixelShuffle3dF32(up, B: B * F, C_out: C, D_in: 1, H_in: H, W_in: W, p1: 1, p2: 2, p3: 2)
        let newH = H * 2, newW = W * 2
        log?("pixel_shuffle: \(f32Stats(up, count: B*F*C*newH*newW))")
        // up is [B*F, C, H*2, W*2] = [F, C, newH*newW]. Transpose back to [C, F, newH*newW] = [B, C, F, H*2, W*2]
        x = transposeHSF32(up, seqLen: C, nHeads: F, headDim: newH * newW)
        pool.reset(keeping: [x])

        // Post-upsample ResBlocks
        for (i, rb) in weights.postResBlocks.enumerated() {
            x = resBlock(x, weights: rb, B: B, C: C, D: F, H: newH, W: newW)
            pool.reset(keeping: [x])
            log?("postResBlock\(i): \(f32Stats(x, count: B*C*F*newH*newW))")
        }

        // Final conv3d: 512 → 128
        x = conv3dF32(x, weight: weights.finalConv.weight, bias: weights.finalConv.bias,
                      B: B, C_in: C, D: F, H: newH, W: newW,
                      C_out: 128, kD: 3, kH: 3, kW: 3, pD: 1, pH: 1, pW: 1)
        pool.reset(keeping: [x])
        log?("final_conv: \(f32Stats(x, count: B*128*F*newH*newW))")
        return x
    }

    private func resBlock(_ x: MTLBuffer, weights w: UpsamplerResBlockWeights,
                          B: Int, C: Int, D: Int, H: Int, W: Int) -> MTLBuffer {
        let spatial = D * H * W
        let count = B * C * spatial
        var h = conv3dF32(x, weight: w.conv1.weight, bias: w.conv1.bias,
                          B: B, C_in: C, D: D, H: H, W: W,
                          C_out: C, kD: 3, kH: 3, kW: 3, pD: 1, pH: 1, pW: 1)
        h = groupNorm3dF32(h, weight: w.norm1Weight, bias: w.norm1Bias,
                           B: B, C: C, spatialSize: spatial, numGroups: 32)
        h = siluF32(h, count: count)
        h = conv3dF32(h, weight: w.conv2.weight, bias: w.conv2.bias,
                      B: B, C_in: C, D: D, H: H, W: W,
                      C_out: C, kD: 3, kH: 3, kW: 3, pD: 1, pH: 1, pW: 1)
        h = groupNorm3dF32(h, weight: w.norm2Weight, bias: w.norm2Bias,
                           B: B, C: C, spatialSize: spatial, numGroups: 32)
        let added = elemAddF32OOP(h, x, count: count)
        return siluF32(added, count: count)
    }
}

// MARK: - AdaIN

/// AdaIN f32: match per-channel mean/std of latents to reference. All f32 MTLBuffers.
public func adainFilter(_ latents: MTLBuffer, reference: MTLBuffer,
                         B: Int, C: Int, latShape: [Int], refShape: [Int],
                         factor: Float = 1.0) -> MTLBuffer {
    let latSpatial = latShape[2] * latShape[3] * latShape[4]
    let refSpatial = refShape[2] * refShape[3] * refShape[4]

    let latPtr = latents.contents().assumingMemoryBound(to: Float.self)
    let refPtr = reference.contents().assumingMemoryBound(to: Float.self)

    var scaleData = [Float](repeating: 0, count: B * C)
    var shiftData = [Float](repeating: 0, count: B * C)

    for b in 0..<B {
        for c in 0..<C {
            var refSum: Float = 0, refSqSum: Float = 0
            let refBase = (b * C + c) * refSpatial
            for s in 0..<refSpatial {
                let v = refPtr[refBase + s]
                refSum += v; refSqSum += v * v
            }
            let refMean = refSum / Float(refSpatial)
            let refVar = refSqSum / Float(refSpatial) - refMean * refMean
            let refStd = sqrtf(max(refVar, 1e-8))

            var latSum: Float = 0, latSqSum: Float = 0
            let latBase = (b * C + c) * latSpatial
            for s in 0..<latSpatial {
                let v = latPtr[latBase + s]
                latSum += v; latSqSum += v * v
            }
            let latMean = latSum / Float(latSpatial)
            let latVar = latSqSum / Float(latSpatial) - latMean * latMean
            let latStd = sqrtf(max(latVar, 1e-8))

            let sc = refStd / latStd
            let sh = refMean - sc * latMean
            let finalScale = 1.0 - factor + factor * sc
            let finalShift = factor * sh
            scaleData[b * C + c] = finalScale
            shiftData[b * C + c] = finalShift
        }
    }

    let scaleBuf = scaleData.withUnsafeBytes {
        MetalContext.shared.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
    }
    let shiftBuf = shiftData.withUnsafeBytes {
        MetalContext.shared.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
    }
    return channelScaleBiasF32(x: latents, scale: scaleBuf, bias: shiftBuf,
                                B: B, C: C, spatial: latSpatial)
}

// MARK: - Weight Loading

public func loadUpsamplerWeights(from url: URL) throws -> UpsamplerWeights {
    let file = try SafeTensorsFile(url: url)
    let ctx = MetalContext.shared

    func loadBufF32(_ name: String) -> MTLBuffer {
        guard let info = file.tensors[name], let ptr = file.pointer(for: name) else {
            fatalError("Upsampler weight not found: \(name)")
        }
        let count = info.shape.reduce(1, *)
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
        fatalError("Unsupported dtype for \(name)")
    }

    func loadTensorF32(_ name: String) -> Tensor {
        guard let info = file.tensors[name], let ptr = file.pointer(for: name) else {
            fatalError("Upsampler weight not found: \(name)")
        }
        let count = info.shape.reduce(1, *)
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
        fatalError("Unsupported dtype for \(name)")
    }

    func loadConv3d(_ prefix: String, C_in: Int, C_out: Int, kD: Int = 3, kH: Int = 3, kW: Int = 3) -> Conv3DWeightF32 {
        let w = loadBufF32("\(prefix).weight")
        let b = loadBufF32("\(prefix).bias")
        return Conv3DWeightF32(weight: w, bias: b, C_in: C_in, C_out: C_out, kD: kD, kH: kH, kW: kW)
    }

    func loadConv2dAsConv3d(_ prefix: String, C_in: Int, C_out: Int) -> Conv3DWeightF32 {
        let w = loadBufF32("\(prefix).weight")
        let b = loadBufF32("\(prefix).bias")
        return Conv3DWeightF32(weight: w, bias: b, C_in: C_in, C_out: C_out, kD: 1, kH: 3, kW: 3)
    }

    func loadResBlock(_ prefix: String, C: Int) -> UpsamplerResBlockWeights {
        return UpsamplerResBlockWeights(
            conv1: loadConv3d("\(prefix).conv1", C_in: C, C_out: C),
            norm1Weight: loadTensorF32("\(prefix).norm1.weight"),
            norm1Bias: loadTensorF32("\(prefix).norm1.bias"),
            conv2: loadConv3d("\(prefix).conv2", C_in: C, C_out: C),
            norm2Weight: loadTensorF32("\(prefix).norm2.weight"),
            norm2Bias: loadTensorF32("\(prefix).norm2.bias")
        )
    }

    let midC = 512

    let initialConv = loadConv3d("initial_conv", C_in: 128, C_out: midC)
    let initialNormW = loadTensorF32("initial_norm.weight")
    let initialNormB = loadTensorF32("initial_norm.bias")

    var resBlocks = [UpsamplerResBlockWeights]()
    for i in 0..<4 { resBlocks.append(loadResBlock("res_blocks.\(i)", C: midC)) }

    let upsampleConv = loadConv2dAsConv3d("upsampler.0", C_in: midC, C_out: midC * 4)

    var postResBlocks = [UpsamplerResBlockWeights]()
    for i in 0..<4 { postResBlocks.append(loadResBlock("post_upsample_res_blocks.\(i)", C: midC)) }

    let finalConv = loadConv3d("final_conv", C_in: midC, C_out: 128)

    return UpsamplerWeights(
        initialConv: initialConv,
        initialNormWeight: initialNormW,
        initialNormBias: initialNormB,
        resBlocks: resBlocks,
        upsampleConv: upsampleConv,
        postResBlocks: postResBlocks,
        finalConv: finalConv,
        midChannels: midC
    )
}
