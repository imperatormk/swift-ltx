// LTX-Video VAE Decoder — pure Swift + Metal.
// Im2col+GEMM for small C_in, gather-GEMM for large C_in. Fits in ~1.4GB on iPhone.
//
// Architecture (from actual safetensors weights):
//   conv_in (128→1024, 3×3×3)
//   up_block 0: UNetMidBlock3D(1024, 5 res_blocks, time_embedder→4096)
//   up_block 1: DepthToSpaceUpsample(conv 1024→4096, stride=2×2×2, residual, /2)
//   up_block 2: UNetMidBlock3D(512, 5 res_blocks, time_embedder→2048)
//   up_block 3: DepthToSpaceUpsample(conv 512→2048, stride=2×2×2, residual, /2)
//   up_block 4: UNetMidBlock3D(256, 5 res_blocks, time_embedder→1024)
//   up_block 5: DepthToSpaceUpsample(conv 256→1024, stride=2×2×2, residual, /2)
//   up_block 6: UNetMidBlock3D(128, 5 res_blocks, time_embedder→512)
//   pixel_norm → ada_ln(last) → silu → conv_out (128→48, 3×3×3)
//   unpatchify (patch_size=4): [B,48,F,H,W] → [B,3,F,H*4,W*4]

import Metal
import Foundation

// MARK: - Weight containers

struct Conv3DWeight {
    let weight: Tensor  // [C_out, C_in, kD, kH, kW] f16
    let bias: Tensor    // [C_out] f16
    let C_in: Int, C_out: Int
    let kD: Int, kH: Int, kW: Int
    let sD: Int, sH: Int, sW: Int
}

struct TimestepEmbedderWeights {
    let linear1Weight: Tensor  // [out, 256] f16
    let linear1Bias: Tensor    // [out] f16
    let linear2Weight: Tensor  // [out, out] f16
    let linear2Bias: Tensor    // [out] f16
    let outDim: Int
}

struct ResBlockWeights {
    let scaleShiftTable: Tensor  // [4, C] f16
    let conv1: Conv3DWeight
    let conv2: Conv3DWeight
    let channels: Int
}

struct UNetMidBlockWeights {
    let timeEmbedder: TimestepEmbedderWeights
    let resBlocks: [ResBlockWeights]
    let channels: Int
}

struct DepthToSpaceWeights {
    let conv: Conv3DWeight
    let stride: (Int, Int, Int)
    let residual: Bool
    let outChannelsReductionFactor: Int
}

// MARK: - VAE Decoder Model

public class VAEDecoder {
    let convIn: Conv3DWeight        // 128 → 1024
    let upBlocks: [Any]             // mix of UNetMidBlockWeights and DepthToSpaceWeights
    let convOut: Conv3DWeight       // 128 → 48
    let patchSize: Int = 4

    // Timestep conditioning
    let timestepScaleMultiplier: Float
    let lastTimeEmbedder: TimestepEmbedderWeights
    let lastScaleShiftTable: Tensor  // [2, 128] f16

    // Per-channel latent statistics for denormalization
    let stdOfMeans: Tensor   // [128] f16
    let meanOfMeans: Tensor  // [128] f16

    public init(file: SafeTensorsFile, prefix: String = "decoder.") throws {
        let ctx = MetalContext.shared

        // Helper: load f16 tensor into Metal buffer
        func load(_ name: String) -> Tensor {
            let key = prefix + name
            guard let info = file.tensors[key] else {
                fatalError("Weight not found: \(key)")
            }
            let ptr = file.pointer(for: key)!
            if info.dtype == .float16 {
                let buf = ctx.device.makeBuffer(bytes: ptr, length: info.byteCount, options: .storageModeShared)!
                return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
            } else if info.dtype == .float32 {
                // Convert f32 → f16
                let count = info.byteCount / 4
                let src = ptr.assumingMemoryBound(to: Float.self)
                let buf = ctx.device.makeBuffer(length: count * 2, options: .storageModeShared)!
                let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<count {
                    dst[i] = VAEDecoder.float32ToFloat16(src[i])
                }
                return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
            } else if info.dtype == .bfloat16 {
                let count = info.byteCount / 2
                let src = ptr.assumingMemoryBound(to: UInt16.self)
                let buf = ctx.device.makeBuffer(length: count * 2, options: .storageModeShared)!
                let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<count {
                    let f = Float(bitPattern: UInt32(src[i]) << 16)
                    dst[i] = VAEDecoder.float32ToFloat16(f)
                }
                return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
            }
            fatalError("Unsupported dtype: \(info.dtype)")
        }

        func loadFloat(_ name: String) -> Float {
            let key = prefix + name
            guard let info = file.tensors[key] else {
                fatalError("Weight not found: \(key)")
            }
            let ptr = file.pointer(for: key)!
            if info.dtype == .float32 {
                return ptr.assumingMemoryBound(to: Float.self).pointee
            } else if info.dtype == .float16 {
                let h = ptr.assumingMemoryBound(to: UInt16.self).pointee
                return VAEDecoder.float16ToFloat32(h)
            }
            fatalError("Unsupported dtype for scalar: \(info.dtype)")
        }

        func loadConv(_ p: String, C_in: Int, C_out: Int, sD: Int = 1, sH: Int = 1, sW: Int = 1) -> Conv3DWeight {
            let w = load("\(p).conv.weight")
            let b = load("\(p).conv.bias")
            return Conv3DWeight(weight: w, bias: b, C_in: C_in, C_out: C_out,
                              kD: 3, kH: 3, kW: 3, sD: sD, sH: sH, sW: sW)
        }

        func loadTimeEmbedder(_ p: String, outDim: Int) -> TimestepEmbedderWeights {
            return TimestepEmbedderWeights(
                linear1Weight: load("\(p).timestep_embedder.linear_1.weight"),
                linear1Bias: load("\(p).timestep_embedder.linear_1.bias"),
                linear2Weight: load("\(p).timestep_embedder.linear_2.weight"),
                linear2Bias: load("\(p).timestep_embedder.linear_2.bias"),
                outDim: outDim
            )
        }

        func loadResBlock(_ p: String, ch: Int) -> ResBlockWeights {
            return ResBlockWeights(
                scaleShiftTable: load("\(p).scale_shift_table"),
                conv1: loadConv("\(p).conv1", C_in: ch, C_out: ch),
                conv2: loadConv("\(p).conv2", C_in: ch, C_out: ch),
                channels: ch
            )
        }

        // conv_in: [1024, 128, 3, 3, 3]
        convIn = loadConv("conv_in", C_in: 128, C_out: 1024)

        // up_blocks: 7 blocks alternating UNetMidBlock3D and DepthToSpaceUpsample
        // Block layout from actual weights:
        //   0: UNetMidBlock3D(1024, 5 res, te→4096)
        //   1: DepthToSpace(1024→4096)  → after shuffle: 512ch, 2x spatial
        //   2: UNetMidBlock3D(512, 5 res, te→2048)
        //   3: DepthToSpace(512→2048)   → after shuffle: 256ch, 2x spatial
        //   4: UNetMidBlock3D(256, 5 res, te→1024)
        //   5: DepthToSpace(256→1024)   → after shuffle: 128ch, 2x spatial
        //   6: UNetMidBlock3D(128, 5 res, te→512)
        var blocks: [Any] = []

        let midBlockConfig: [(idx: Int, ch: Int, teDim: Int, numRes: Int)] = [
            (0, 1024, 4096, 5),
            (2, 512, 2048, 5),
            (4, 256, 1024, 5),
            (6, 128, 512, 5),
        ]

        let dtsBlockConfig: [(idx: Int, C_in: Int, C_out: Int)] = [
            (1, 1024, 4096),
            (3, 512, 2048),
            (5, 256, 1024),
        ]

        // Load all 7 blocks in order
        for i in 0..<7 {
            if let mid = midBlockConfig.first(where: { $0.idx == i }) {
                let te = loadTimeEmbedder("up_blocks.\(i).time_embedder", outDim: mid.teDim)
                var resBlocks: [ResBlockWeights] = []
                for r in 0..<mid.numRes {
                    resBlocks.append(loadResBlock("up_blocks.\(i).res_blocks.\(r)", ch: mid.ch))
                }
                blocks.append(UNetMidBlockWeights(timeEmbedder: te, resBlocks: resBlocks, channels: mid.ch))
            } else if let dts = dtsBlockConfig.first(where: { $0.idx == i }) {
                let conv = loadConv("up_blocks.\(i).conv", C_in: dts.C_in, C_out: dts.C_out)
                blocks.append(DepthToSpaceWeights(conv: conv, stride: (2,2,2), residual: true, outChannelsReductionFactor: 2))
            }
        }

        upBlocks = blocks

        // conv_out: [48, 128, 3, 3, 3]
        convOut = loadConv("conv_out", C_in: 128, C_out: 48)

        // Timestep conditioning
        timestepScaleMultiplier = loadFloat("timestep_scale_multiplier")
        lastTimeEmbedder = loadTimeEmbedder("last_time_embedder", outDim: 256)
        lastScaleShiftTable = load("last_scale_shift_table")

        // Load per-channel latent statistics (keys use "vae.per_channel_statistics." prefix, not "decoder.")
        let statsPrefix: String
        if prefix == "decoder." {
            statsPrefix = "vae."
        } else if prefix == "vae.decoder." {
            statsPrefix = "vae."
        } else {
            statsPrefix = ""
        }
        func loadStats(_ name: String) -> Tensor {
            let key = statsPrefix + "per_channel_statistics." + name
            guard let info = file.tensors[key] else {
                fatalError("Stats not found: \(key)")
            }
            let ptr = file.pointer(for: key)!
            if info.dtype == .float16 {
                let buf = ctx.device.makeBuffer(bytes: ptr, length: info.byteCount, options: .storageModeShared)!
                return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
            } else if info.dtype == .float32 {
                let count = info.byteCount / 4
                let src = ptr.assumingMemoryBound(to: Float.self)
                let buf = ctx.device.makeBuffer(length: count * 2, options: .storageModeShared)!
                let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<count {
                    dst[i] = VAEDecoder.float32ToFloat16(src[i])
                }
                return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
            }
            fatalError("Unsupported dtype for stats: \(info.dtype)")
        }
        stdOfMeans = loadStats("std-of-means")
        meanOfMeans = loadStats("mean-of-means")

    }

    // MARK: - Decode

    /// Decode latent [1, 128, F_lat, H_lat, W_lat] → image [1, 3, F_out, H_out, W_out].
    /// timestep: typically 0.05 for inference (matching MLX reference).
    public func decode(_ latent: Tensor, timestep: Float = 0.05, debug: Bool = false, log: ((String) -> Void)? = nil) -> Tensor {
        let ctx = MetalContext.shared
        let pool = ctx.bufferPool

        let B = latent.shape[0]
        let C_in = latent.shape[1]  // 128
        let F_lat = latent.shape[2], H_lat = latent.shape[3], W_lat = latent.shape[4]

        var _debugLog = ""
        let t0 = CFAbsoluteTimeGetCurrent()
        var tLast = t0
        func step(_ msg: String) {
            let now = CFAbsoluteTimeGetCurrent()
            let dt = now - tLast
            let elapsed = now - t0
            tLast = now
            let line = String(format: "[%.2fs +%.3fs] %@", elapsed, dt, msg)
            log?(line)
            _debugLog += line + "\n"
            try? _debugLog.write(toFile: "/Users/zimski/projects/oss/swift-llm/decode_log.txt", atomically: true, encoding: .utf8)
        }

        // Compute scaled timestep
        let scaledTimestep = timestep * timestepScaleMultiplier
        step("timestep=\(timestep) scale_mult=\(timestepScaleMultiplier) scaled=\(scaledTimestep)")

        // Create timestep tensor [B] f32
        var tsData = [Float](repeating: scaledTimestep, count: B)
        let tsBuf = ctx.makeBuffer(&tsData, length: B * 4)
        let tsTensor = Tensor(buffer: tsBuf, shape: [B], dtype: .float32)

        // Helper: dump tensor stats (reads back GPU data)
        func tensorStats(_ t: Tensor, label: String) -> String {
            let count = t.count
            let ptr = t.buffer.contents().assumingMemoryBound(to: Float16.self)
            var mn: Float = .infinity, mx: Float = -.infinity, sum: Float = 0
            var nanCount = 0, infCount = 0, zeroCount = 0
            for i in 0..<count {
                let v = Float(ptr[i])
                if v.isNaN { nanCount += 1; continue }
                if v.isInfinite { infCount += 1; continue }
                if v == 0 { zeroCount += 1 }
                mn = min(mn, v); mx = max(mx, v); sum += v
            }
            let mean = sum / Float(max(count - nanCount - infCount, 1))
            return "\(label): min=\(String(format:"%.4f",mn)) max=\(String(format:"%.4f",mx)) mean=\(String(format:"%.6f",mean)) nan=\(nanCount) inf=\(infCount) zero=\(zeroCount)/\(count)"
        }

        // NOTE: latents from pipeline are already denormalized + noised.
        // Denormalization (latent * std_of_means + mean_of_means) happens before saving.

        // conv_in: causal pad + conv3d (128 → 1024)
        step("conv_in: \(C_in)→1024, 3x3x3, spatial \(F_lat)x\(H_lat)x\(W_lat)")
        if debug { step(tensorStats(latent, label: "latent_in")) }
        var x = causalConv3d(latent, conv: convIn, B: B,
                            C_in: C_in, D: F_lat, H: H_lat, W: W_lat, causal: false)
        pool.reset(keeping: [x.buffer])
        var C = 1024
        var D = F_lat, H = H_lat, W = W_lat
        if debug { step(tensorStats(x, label: "conv_in_out")) }
        step("conv_in done, shape [\(B),\(C),\(D),\(H),\(W)]")

        // Process up_blocks
        for (i, block) in upBlocks.enumerated() {
            if let midBlock = block as? UNetMidBlockWeights {
                step("up_block \(i): MidBlock(\(C), \(midBlock.resBlocks.count) res), spatial \(D)x\(H)x\(W)")
                x = runMidBlock(x, weights: midBlock, B: B, C: C, D: D, H: H, W: W,
                              timestep: tsTensor, scaledTimestep: scaledTimestep)
                pool.reset(keeping: [x.buffer]); pool.purge()
                if debug { step(tensorStats(x, label: "up_block_\(i)_out")) }
            } else if let dtsBlock = block as? DepthToSpaceWeights {
                step("up_block \(i): DepthToSpace(\(C)→\(dtsBlock.conv.C_out)), spatial \(D)x\(H)x\(W)")
                x = runDepthToSpaceUpsample(x, weights: dtsBlock, B: B, C: C, D: D, H: H, W: W, causal: false)
                let (p1, p2, p3) = dtsBlock.stride
                let newC = dtsBlock.conv.C_out / (p1 * p2 * p3)
                if p1 == 2 {
                    D = D * p1 - 1
                } else {
                    D = D * p1
                }
                H = H * p2
                W = W * p3
                C = newC
                pool.reset(keeping: [x.buffer]); pool.purge()
                if debug { step(tensorStats(x, label: "up_block_\(i)_out")) }
            }
        }

        // Tail ops: pixnorm → adaln → silu (batched together)
        step("pixel_norm C=\(C), spatial \(D)x\(H)x\(W)")
        let spatialSize = D * H * W
        ctx.beginBatch()
        x = pixelNorm3d(x, B: B, C: C, spatialSize: spatialSize)
        x = applyLastAdaLN(x, B: B, C: C, spatialSize: spatialSize,
                          timestep: tsTensor, scaledTimestep: scaledTimestep)
        x = silu(x)
        ctx.endBatch()
        if debug {
            step(tensorStats(x, label: "after_silu"))
        }
        step("last AdaLN + silu done")

        // conv_out
        step("conv_out: \(C)→48, 3x3x3, spatial \(D)x\(H)x\(W)")
        pool.reset(keeping: [x.buffer]); pool.purge()
        x = causalConv3d(x, conv: convOut, B: B, C_in: C, D: D, H: H, W: W, causal: false)
        pool.reset(keeping: [x.buffer]); pool.purge()
        if debug { step(tensorStats(x, label: "after_conv_out")) }

        // unpatchify: [B, 48, F, H, W] → [B, 3, F, H*4, W*4]
        step("unpatchify")
        x = unpatchify3d(x, B: B, C_out: 3, F: D, H_in: H, W_in: W, patchSize: patchSize)
        pool.reset(keeping: [x.buffer]); pool.purge()
        if debug { step(tensorStats(x, label: "final_output")) }

        log?("decode complete, shape \(x.shape)")
        return x
    }

    // MARK: - Block execution

    private func causalConv3d(_ x: Tensor, conv: Conv3DWeight, B: Int,
                             C_in: Int, D: Int, H: Int, W: Int, causal: Bool) -> Tensor {
        // Causal padding: repeat first frame (time_kernel_size - 1) times at front
        // Non-causal padding: repeat first frame (ks-1)/2 times, last frame (ks-1)/2 times
        let timePad: Int
        var padded = x
        var D_padded = D
        if causal {
            timePad = conv.kD - 1
            if timePad > 0 {
                padded = causalPad(x, B: B, C: C_in, D: D, H: H, W: W, pad: timePad)
                D_padded = D + timePad
            }
        } else {
            // Non-causal: pad (kD-1)/2 on each side by repeating boundary frames
            let halfPad = (conv.kD - 1) / 2
            if halfPad > 0 {
                // First: causal pad at front
                padded = causalPad(x, B: B, C: C_in, D: D, H: H, W: W, pad: halfPad)
                D_padded = D + halfPad
                // Then: pad at back by reversing + causal pad + reversing
                // Simpler: just use padding parameter directly. The conv already handles spatial padding.
                // For temporal: we've added halfPad frames at front. We need halfPad at back too.
                // Use another causal pad trick: reverse time, pad, reverse.
                // Actually, for non-causal with replicate padding at both ends, let's just do it on CPU for now.
                // The padded tensor has shape [B, C, D+halfPad, H, W].
                // We need to also pad halfPad at the end. Let's extend causal_pad to handle both sides.
                // For simplicity, add the back-padding by appending copies of the last frame.
                // We can reuse causal_pad by flipping. Or just handle it in the kernel with spatial padding.
                //
                // Actually, the CausalConv3d in PyTorch only pads temporally. Spatial padding is handled
                // by conv3d's padding parameter. For non-causal mode, it pads (kD-1)/2 on each side.
                // Let's pad the back side too:
                padded = causalPadBack(padded, B: B, C: C_in, D: D_padded, H: H, W: W, pad: halfPad)
                D_padded = D_padded + halfPad
            }
        }

        // Spatial padding is built into the conv3d kernel via pH, pW
        let spatialPad = conv.kH / 2  // kernel_size=3 → padding=1
        return conv3d(padded, weight: conv.weight, bias: conv.bias,
                     B: B, C_in: C_in, D: D_padded, H: H, W: W,
                     C_out: conv.C_out, kD: conv.kD, kH: conv.kH, kW: conv.kW,
                     sD: conv.sD, sH: conv.sH, sW: conv.sW,
                     pD: 0, pH: spatialPad, pW: spatialPad)
    }

    // causalPadBack is now a GPU kernel in Ops.swift — no CPU memcpy needed

    private func runMidBlock(_ x: Tensor, weights: UNetMidBlockWeights, B: Int, C: Int, D: Int, H: Int, W: Int,
                            timestep: Tensor, scaledTimestep: Float) -> Tensor {
        // Compute timestep embedding: sinusoidal(256) → linear1 → silu → linear2
        let tsEmb = computeTimestepEmbedding(timestep, embedder: weights.timeEmbedder, B: B,
                                             scaledTimestep: scaledTimestep)
        // tsEmb: [B, outDim] f16 → reshape to [B, outDim, 1, 1, 1] for broadcasting
        // outDim = channels * 4 for mid blocks

        let pool = MetalContext.shared.bufferPool
        var h = x
        for resBlock in weights.resBlocks {
            h = runResBlock(h, weights: resBlock, B: B, C: C, D: D, H: H, W: W, tsEmb: tsEmb)
            pool.reset(keeping: [h.buffer, tsEmb.buffer])
            pool.purge()
        }
        return h
    }

    private func runResBlock(_ x: Tensor, weights: ResBlockWeights, B: Int, C: Int, D: Int, H: Int, W: Int,
                            tsEmb: Tensor) -> Tensor {
        let pool = MetalContext.shared.bufferPool
        let spatialSize = D * H * W
        var h = x

        // norm1 (pixel_norm)
        h = pixelNorm3d(h, B: B, C: C, spatialSize: spatialSize)

        // AdaLN: compute all 4 scale/shift values, then apply
        let (scale1, shift1, scale2, shift2) = computeAdaLN(tsEmb, table: weights.scaleShiftTable, B: B, C: C)
        h = scaleShift3d(h, scale: scale1, shift: shift1, C: C, spatialSize: spatialSize)

        h = silu(h)

        // Free pre-conv intermediates, keep x (for residual), h (conv input), scale2/shift2/tsEmb (for later)
        pool.reset(keeping: [x.buffer, h.buffer, scale2.buffer, shift2.buffer, tsEmb.buffer]); pool.purge()

        // conv1
        h = causalConv3d(h, conv: weights.conv1, B: B, C_in: C, D: D, H: H, W: W, causal: false)
        pool.reset(keeping: [x.buffer, h.buffer, scale2.buffer, shift2.buffer, tsEmb.buffer]); pool.purge()

        // norm2 + adaln2 + silu
        h = pixelNorm3d(h, B: B, C: C, spatialSize: spatialSize)
        h = scaleShift3d(h, scale: scale2, shift: shift2, C: C, spatialSize: spatialSize)
        h = silu(h)

        // Free pre-conv2 intermediates
        pool.reset(keeping: [x.buffer, h.buffer, tsEmb.buffer]); pool.purge()

        // conv2
        h = causalConv3d(h, conv: weights.conv2, B: B, C_in: C, D: D, H: H, W: W, causal: false)
        pool.reset(keeping: [x.buffer, h.buffer, tsEmb.buffer]); pool.purge()

        // residual
        h = elemAdd(x, h)
        return h
    }

    private func runDepthToSpaceUpsample(_ x: Tensor, weights: DepthToSpaceWeights, B: Int,
                                        C: Int, D: Int, H: Int, W: Int, causal: Bool) -> Tensor {
        let pool = MetalContext.shared.bufferPool
        let (p1, p2, p3) = weights.stride
        let prodStride = p1 * p2 * p3  // 8

        var result: Tensor

        if weights.residual {
            let skipC = C / prodStride
            let numRepeat = prodStride / weights.outChannelsReductionFactor
            let newD = D * p1
            let newH = H * p2
            let newW = W * p3

            // Skip path first: pixel_shuffle(x) → repeat → temporal slice
            var x_repeated = pixelShuffle3d(x, B: B, C_out: skipC, D_in: D, H_in: H, W_in: W,
                                           p1: p1, p2: p2, p3: p3)
            if numRepeat > 1 {
                let tmp = x_repeated
                x_repeated = repeatChannels(tmp, B: B, C: skipC, spatialSize: newD * newH * newW, repeats: numRepeat)
                // tmp can be freed now
            }
            if p1 == 2 {
                let tmp = x_repeated
                x_repeated = temporalSlice(tmp, B: B, C: skipC * numRepeat, D: newD, H: newH, W: newW, from: 1)
            }
            // Free intermediates, keep x (needed for conv) and x_repeated (needed for add)
            pool.reset(keeping: [x.buffer, x_repeated.buffer]); pool.purge()

            // Conv path: conv → pixel_shuffle → temporal slice
            var h = causalConv3d(x, conv: weights.conv, B: B, C_in: C, D: D, H: H, W: W, causal: causal)
            // x no longer needed
            pool.reset(keeping: [h.buffer, x_repeated.buffer]); pool.purge()

            let convOutC = weights.conv.C_out
            let tmp = h
            h = pixelShuffle3d(tmp, B: B, C_out: convOutC / prodStride, D_in: D, H_in: H, W_in: W,
                              p1: p1, p2: p2, p3: p3)
            pool.reset(keeping: [h.buffer, x_repeated.buffer]); pool.purge()

            if p1 == 2 {
                let tmp2 = h
                h = temporalSlice(tmp2, B: B, C: convOutC / prodStride, D: newD, H: newH, W: newW, from: 1)
                pool.reset(keeping: [h.buffer, x_repeated.buffer]); pool.purge()
            }

            result = elemAdd(h, x_repeated)
        } else {
            var h = causalConv3d(x, conv: weights.conv, B: B, C_in: C, D: D, H: H, W: W, causal: causal)
            pool.reset(keeping: [h.buffer]); pool.purge()
            let convOutC = weights.conv.C_out
            h = pixelShuffle3d(h, B: B, C_out: convOutC / prodStride, D_in: D, H_in: H, W_in: W,
                              p1: p1, p2: p2, p3: p3)
            if p1 == 2 {
                let newD = D * p1
                h = temporalSlice(h, B: B, C: convOutC / prodStride, D: newD, H: H * p2, W: W * p3, from: 1)
            }
            result = h
        }
        return result
    }

    // MARK: - Timestep embedding helpers

    private func computeTimestepEmbedding(_ timestep: Tensor, embedder: TimestepEmbedderWeights, B: Int,
                                         scaledTimestep: Float) -> Tensor {
        // Sinusoidal embedding: [B] → [B, 256]
        let sinEmb = timestepEmbedding(timestep, B: B, dim: 256)
        // MLP: linear1 → silu → linear2
        var h = linearF16(sinEmb, weight: embedder.linear1Weight, bias: embedder.linear1Bias,
                         B: B, K: 256, N: embedder.outDim)
        h = silu(h)
        h = linearF16(h, weight: embedder.linear2Weight, bias: embedder.linear2Bias,
                     B: B, K: embedder.outDim, N: embedder.outDim)
        return h  // [B, outDim]
    }

    /// Compute AdaLN scale/shift from timestep embedding (GPU).
    /// table: [4, C] f16, tsEmb: [B, C*4] f16
    /// Returns (scale1, shift1, scale2, shift2) each [B, C] f16.
    private func computeAdaLN(_ tsEmb: Tensor, table: Tensor, B: Int, C: Int) -> (Tensor, Tensor, Tensor, Tensor) {
        let (shift1, scale1, shift2, scale2) = adaLNCombine4(tsEmb, table: table, B: B, C: C)
        return (scale1, shift1, scale2, shift2)
    }

    /// Apply last AdaLN before conv_out (GPU).
    /// table: [2, C], timestep embedding through last_time_embedder.
    private func applyLastAdaLN(_ x: Tensor, B: Int, C: Int, spatialSize: Int,
                                timestep: Tensor, scaledTimestep: Float) -> Tensor {
        let tsEmb = computeTimestepEmbedding(timestep, embedder: lastTimeEmbedder, B: B,
                                            scaledTimestep: scaledTimestep)
        let (shift, scale) = adaLNCombine2(tsEmb, table: lastScaleShiftTable, B: B, C: C)
        return scaleShift3d(x, scale: scale, shift: shift, C: C, spatialSize: spatialSize)
    }

    // MARK: - Utility ops

    /// Repeat channels (GPU): [B, C, spatial] → [B, C*repeats, spatial]
    private func repeatChannels(_ x: Tensor, B: Int, C: Int, spatialSize: Int, repeats: Int) -> Tensor {
        return repeatChannelsGPU(x, B: B, C: C, spatialSize: spatialSize, repeats: repeats)
    }

    /// Slice temporal dim (GPU): x[:, :, from:, :, :]. Returns [B, C, D-from, H, W].
    private func temporalSlice(_ x: Tensor, B: Int, C: Int, D: Int, H: Int, W: Int, from: Int) -> Tensor {
        return temporalSliceGPU(x, B: B, C: C, D: D, H: H, W: W, from: from)
    }

    // MARK: - Float16 conversion utilities

    static func float32ToFloat16(_ f: Float) -> UInt16 {
        let bits = f.bitPattern
        let sign = (bits >> 16) & 0x8000
        let exp = Int((bits >> 23) & 0xFF) - 127
        let frac = bits & 0x7FFFFF

        if exp > 15 {
            return UInt16(sign | 0x7C00)  // Inf
        } else if exp < -14 {
            return UInt16(sign)  // Zero/denorm
        }
        let h_exp = UInt16(exp + 15)
        let h_frac = UInt16(frac >> 13)
        return UInt16(sign) | (h_exp << 10) | h_frac
    }

    static func float16ToFloat32(_ h: UInt16) -> Float {
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
}
