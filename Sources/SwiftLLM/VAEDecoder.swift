// LTX-Video VAE Decoder — pure Swift + Metal.
// Set to true to enable per-tile debug logs + [MEM] stats in perf callback (adds overhead).
private let vaeDebugLogs = true

// MARK: - Memory config
// Tile target bytes for each tiled operation. Raise on devices with more RAM → fewer tiles → faster.
// WARNING: resblock peak is always 2× full activation (x + conv1Out both full-D) regardless of tile size.
// For block 6 at C=128, D=73, H=160, W=160: full activation = 478MB, peak ≥ 956MB + overhead.
private let vaeResTileBytes  = 50 * 1024 * 1024   // resblock P1/P2 D-tile target
private let vaeDtsTileBytes  = 50 * 1024 * 1024   // DepthToSpace conv tile target
private let vaeTailTileBytes = 50 * 1024 * 1024   // tail pixnorm/adaln tile target

private func memTag() -> String {
    guard vaeDebugLogs else { return "" }
    let s = PeakMemoryTracker.shared.sample()
    return " [MEM] Metal=\(MetalContext.shared.device.currentAllocatedSize/1048576)MB Proc=\(s.current/1048576)MB Peak=\(s.peak/1048576)MB"
}
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

public struct Conv3DWeight {
    public let weight: Tensor  // [C_out, C_in, kD, kH, kW] f16
    public let bias: Tensor    // [C_out] f16
    public let C_in: Int, C_out: Int
    public let kD: Int, kH: Int, kW: Int
    public let sD: Int, sH: Int, sW: Int
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

// MARK: - Latent Statistics (lightweight, ~1KB)

/// Per-channel mean/std for latent (de)normalization. Tiny — load once, reuse everywhere.
public struct VAELatentStats {
    public let stdOfMeans: Tensor      // [128] f16
    public let meanOfMeans: Tensor     // [128] f16
    public let stdOfMeansF32: MTLBuffer   // [128] f32
    public let meanOfMeansF32: MTLBuffer  // [128] f32

    public static func load(file: SafeTensorsFile, prefix: String) -> VAELatentStats {
        let statsPrefix: String
        if prefix == "decoder." {
            statsPrefix = "vae."
        } else if prefix == "vae.decoder." {
            statsPrefix = "vae."
        } else {
            statsPrefix = ""
        }
        let ctx = MetalContext.shared
        func loadF16(_ name: String) -> Tensor {
            let key = statsPrefix + "per_channel_statistics." + name
            guard let info = file.tensors[key] else { fatalError("Stats not found: \(key)") }
            let ptr = file.pointer(for: key)!
            if info.dtype == .float16 {
                let buf = ctx.device.makeBuffer(bytes: ptr, length: info.byteCount, options: .storageModeShared)!
                return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
            } else if info.dtype == .float32 {
                let count = info.byteCount / 4
                let src = ptr.assumingMemoryBound(to: Float.self)
                let buf = ctx.device.makeBuffer(length: count * 2, options: .storageModeShared)!
                let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<count { dst[i] = VAEDecoder.float32ToFloat16(src[i]) }
                return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
            }
            fatalError("Unsupported dtype for stats: \(info.dtype)")
        }
        func loadF32(_ name: String) -> MTLBuffer {
            let key = statsPrefix + "per_channel_statistics." + name
            guard let info = file.tensors[key], let ptr = file.pointer(for: key) else { fatalError("Stats not found: \(key)") }
            let count = info.shape.reduce(1, *)
            if info.dtype == .float32 {
                return ctx.device.makeBuffer(bytes: ptr, length: count * 4, options: .storageModeShared)!
            } else if info.dtype == .float16 {
                let src = ptr.assumingMemoryBound(to: UInt16.self)
                let buf = ctx.device.makeBuffer(length: count * 4, options: .storageModeShared)!
                let dst = buf.contents().assumingMemoryBound(to: Float.self)
                for i in 0..<count { dst[i] = Float(Float16(bitPattern: src[i])) }
                return buf
            }
            fatalError("Unsupported dtype for f32 stats: \(info.dtype)")
        }
        return VAELatentStats(
            stdOfMeans: loadF16("std-of-means"),
            meanOfMeans: loadF16("mean-of-means"),
            stdOfMeansF32: loadF32("std-of-means"),
            meanOfMeansF32: loadF32("mean-of-means")
        )
    }

    /// Denormalize: out = latent * std + mean (f32)
    public func denormalize(_ buf: MTLBuffer, B: Int, C: Int, F: Int, H: Int, W: Int) -> MTLBuffer {
        channelScaleBiasF32(x: buf, scale: stdOfMeansF32, bias: meanOfMeansF32,
                            B: B, C: C, spatial: F * H * W)
    }

    /// Normalize: out = (latent - mean) / std (f32)
    public func normalize(_ buf: MTLBuffer, B: Int, C: Int, F: Int, H: Int, W: Int) -> MTLBuffer {
        let stdPtr = stdOfMeansF32.contents().assumingMemoryBound(to: Float.self)
        let meanPtr = meanOfMeansF32.contents().assumingMemoryBound(to: Float.self)
        var scaleData = [Float](repeating: 0, count: C)
        var biasData = [Float](repeating: 0, count: C)
        for c in 0..<C {
            let s = stdPtr[c]
            let m = meanPtr[c]
            let invS = s != 0 ? 1.0 / s : 0.0
            scaleData[c] = invS
            biasData[c] = -m * invS
        }
        let ctx = MetalContext.shared
        let scaleBuf = scaleData.withUnsafeBytes { ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        let biasBuf = biasData.withUnsafeBytes { ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)! }
        return channelScaleBiasF32(x: buf, scale: scaleBuf, bias: biasBuf,
                                    B: B, C: C, spatial: F * H * W)
    }
}

// MARK: - VAE Decoder Model

/// Describes a block to load on demand during decode.
enum VAEBlockDesc {
    case midBlock(idx: Int, ch: Int, teDim: Int, numRes: Int)
    case depthToSpace(idx: Int, C_in: Int, C_out: Int)
}

public class VAEDecoder: @unchecked Sendable {
    let convIn: Conv3DWeight        // 128 → 1024
    let blockDescs: [VAEBlockDesc]  // lazy — loaded on demand during decode
    let convOut: Conv3DWeight       // 128 → 48
    let patchSize: Int = 4

    // Stored for lazy block loading
    let file: SafeTensorsFile
    let prefix: String

    // Timestep conditioning
    let timestepScaleMultiplier: Float
    let lastTimeEmbedder: TimestepEmbedderWeights
    let lastScaleShiftTable: Tensor  // [2, 128] f16

    // Per-channel latent statistics for denormalization
    public let stats: VAELatentStats

    public init(file: SafeTensorsFile, prefix: String = "decoder.", stats: VAELatentStats? = nil) throws {
        self.file = file
        self.prefix = prefix

        // conv_in: [1024, 128, 3, 3, 3] — small, keep resident
        convIn = VAEDecoder.loadConv(file: file, prefix: prefix, name: "conv_in", C_in: 128, C_out: 1024)

        // up_blocks: 7 blocks — descriptors only, weights loaded on demand during decode
        blockDescs = [
            .midBlock(idx: 0, ch: 1024, teDim: 4096, numRes: 5),
            .depthToSpace(idx: 1, C_in: 1024, C_out: 4096),
            .midBlock(idx: 2, ch: 512, teDim: 2048, numRes: 5),
            .depthToSpace(idx: 3, C_in: 512, C_out: 2048),
            .midBlock(idx: 4, ch: 256, teDim: 1024, numRes: 5),
            .depthToSpace(idx: 5, C_in: 256, C_out: 1024),
            .midBlock(idx: 6, ch: 128, teDim: 512, numRes: 5),
        ]

        // conv_out: [48, 128, 3, 3, 3] — small, keep resident
        convOut = VAEDecoder.loadConv(file: file, prefix: prefix, name: "conv_out", C_in: 128, C_out: 48)

        // Timestep conditioning — tiny, keep resident
        timestepScaleMultiplier = VAEDecoder.loadScalar(file: file, prefix: prefix, name: "timestep_scale_multiplier")
        lastTimeEmbedder = VAEDecoder.loadTimeEmbedder(file: file, prefix: prefix, name: "last_time_embedder", outDim: 256)
        lastScaleShiftTable = VAEDecoder.loadTensor(file: file, prefix: prefix, name: "last_scale_shift_table")

        // Per-channel latent statistics — reuse if pre-loaded, otherwise load now
        self.stats = stats ?? VAELatentStats.load(file: file, prefix: prefix)
    }

    // MARK: - Static weight loading helpers (used by init and lazy block loading)

    static func loadTensor(file: SafeTensorsFile, prefix: String, name: String) -> Tensor {
        let key = prefix + name
        guard let info = file.tensors[key] else { fatalError("Weight not found: \(key)") }
        let ptr = file.pointer(for: key)!
        let ctx = MetalContext.shared
        if info.dtype == .float16 {
            let buf = ctx.device.makeBuffer(bytes: ptr, length: info.byteCount, options: .storageModeShared)!
            return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
        } else if info.dtype == .float32 {
            let count = info.byteCount / 4
            let src = ptr.assumingMemoryBound(to: Float.self)
            let buf = ctx.device.makeBuffer(length: count * 2, options: .storageModeShared)!
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<count { dst[i] = float32ToFloat16(src[i]) }
            return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
        } else if info.dtype == .bfloat16 {
            let count = info.byteCount / 2
            let src = ptr.assumingMemoryBound(to: UInt16.self)
            let buf = ctx.device.makeBuffer(length: count * 2, options: .storageModeShared)!
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<count {
                let f = Float(bitPattern: UInt32(src[i]) << 16)
                dst[i] = float32ToFloat16(f)
            }
            return Tensor(buffer: buf, shape: info.shape, dtype: .float16)
        }
        if info.dtype == .int32 {
            // NF4 quantized tensor — dequant to f16
            let suffix = ".weight"
            let baseKey = String(key.prefix(key.count - suffix.count))
            return dequantNF4ToF16(file: file, key: key, info: info,
                                    scalesKey: baseKey + ".scales",
                                    biasesKey: baseKey + ".biases")
        }
        fatalError("Unsupported dtype: \(info.dtype) for \(key)")
    }

    static func loadScalar(file: SafeTensorsFile, prefix: String, name: String) -> Float {
        let key = prefix + name
        guard let info = file.tensors[key] else { fatalError("Weight not found: \(key)") }
        let ptr = file.pointer(for: key)!
        if info.dtype == .float32 { return ptr.assumingMemoryBound(to: Float.self).pointee }
        if info.dtype == .float16 { return float16ToFloat32(ptr.assumingMemoryBound(to: UInt16.self).pointee) }
        fatalError("Unsupported dtype for scalar: \(info.dtype)")
    }

    static func loadConv(file: SafeTensorsFile, prefix: String, name: String, C_in: Int, C_out: Int, sD: Int = 1, sH: Int = 1, sW: Int = 1) -> Conv3DWeight {
        let weightKey = prefix + "\(name).conv.weight"
        let weightInfo = file.tensors[weightKey]!
        let w: Tensor
        if weightInfo.dtype == .int32 {
            // NF4 quantized: dequant to f16 on load
            w = dequantNF4ToF16(file: file, key: weightKey, info: weightInfo,
                                scalesKey: prefix + "\(name).conv.scales",
                                biasesKey: prefix + "\(name).conv.biases")
        } else {
            w = loadTensor(file: file, prefix: prefix, name: "\(name).conv.weight")
        }
        let b = loadTensor(file: file, prefix: prefix, name: "\(name).conv.bias")
        return Conv3DWeight(weight: w, bias: b, C_in: C_in, C_out: C_out,
                           kD: 3, kH: 3, kW: 3, sD: sD, sH: sH, sW: sW)
    }

    /// Dequantize NF4 packed int32 → f16 tensor.
    /// Weight: [N, K/8] int32 (8 nibbles per int32), Scales/Biases: [N, K/groupSize] f16.
    /// Output: [N, K] f16 where K = cols from original shape.
    private static let nf4Table: [Float] = [
        -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
        -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
        0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
    ]

    private static func dequantNF4ToF16(file: SafeTensorsFile, key: String,
                                         info: TensorInfo,
                                         scalesKey: String, biasesKey: String) -> Tensor {
        let N = info.shape[0]
        let packedCols = info.shape[1]  // K/8
        let K = packedCols * 8

        let weightPtr = file.pointer(for: key)!.assumingMemoryBound(to: UInt32.self)
        let scalesPtr = file.pointer(for: scalesKey)!.assumingMemoryBound(to: UInt16.self)
        let biasesPtr = file.pointer(for: biasesKey)!.assumingMemoryBound(to: UInt16.self)

        let scalesInfo = file.tensors[scalesKey]!
        let groupSize = K / scalesInfo.shape[1]

        let ctx = MetalContext.shared
        let buf = ctx.device.makeBuffer(length: N * K * 2, options: .storageModeShared)!
        let dst = buf.contents().assumingMemoryBound(to: UInt16.self)

        for row in 0..<N {
            for packedCol in 0..<packedCols {
                let packed = weightPtr[row * packedCols + packedCol]
                for nib in 0..<8 {
                    let col = packedCol * 8 + nib
                    let idx = Int((packed >> (nib * 4)) & 0xF)
                    let group = col / groupSize
                    let scale = Float(Float16(bitPattern: scalesPtr[row * scalesInfo.shape[1] + group]))
                    let bias = Float(Float16(bitPattern: biasesPtr[row * scalesInfo.shape[1] + group]))
                    let val = nf4Table[idx] * scale + bias
                    dst[row * K + col] = Float16(val).bitPattern
                }
            }
        }
        return Tensor(buffer: buf, shape: [N, K], dtype: .float16)
    }

    static func loadTimeEmbedder(file: SafeTensorsFile, prefix: String, name: String, outDim: Int) -> TimestepEmbedderWeights {
        return TimestepEmbedderWeights(
            linear1Weight: loadTensor(file: file, prefix: prefix, name: "\(name).timestep_embedder.linear_1.weight"),
            linear1Bias: loadTensor(file: file, prefix: prefix, name: "\(name).timestep_embedder.linear_1.bias"),
            linear2Weight: loadTensor(file: file, prefix: prefix, name: "\(name).timestep_embedder.linear_2.weight"),
            linear2Bias: loadTensor(file: file, prefix: prefix, name: "\(name).timestep_embedder.linear_2.bias"),
            outDim: outDim
        )
    }

    static func loadResBlock(file: SafeTensorsFile, prefix: String, name: String, ch: Int) -> ResBlockWeights {
        return ResBlockWeights(
            scaleShiftTable: loadTensor(file: file, prefix: prefix, name: "\(name).scale_shift_table"),
            conv1: loadConv(file: file, prefix: prefix, name: "\(name).conv1", C_in: ch, C_out: ch),
            conv2: loadConv(file: file, prefix: prefix, name: "\(name).conv2", C_in: ch, C_out: ch),
            channels: ch
        )
    }

    /// Load a MidBlock's weights on demand.
    static func loadMidBlock(file: SafeTensorsFile, prefix: String, idx: Int, ch: Int, teDim: Int, numRes: Int) -> UNetMidBlockWeights {
        let p = "up_blocks.\(idx)"
        let te = loadTimeEmbedder(file: file, prefix: prefix, name: "\(p).time_embedder", outDim: teDim)
        var resBlocks: [ResBlockWeights] = []
        for r in 0..<numRes {
            resBlocks.append(loadResBlock(file: file, prefix: prefix, name: "\(p).res_blocks.\(r)", ch: ch))
        }
        return UNetMidBlockWeights(timeEmbedder: te, resBlocks: resBlocks, channels: ch)
    }

    /// Load a DepthToSpace block's weights on demand.
    static func loadDTSBlock(file: SafeTensorsFile, prefix: String, idx: Int, C_in: Int, C_out: Int) -> DepthToSpaceWeights {
        let conv = loadConv(file: file, prefix: prefix, name: "up_blocks.\(idx).conv", C_in: C_in, C_out: C_out)
        return DepthToSpaceWeights(conv: conv, stride: (2,2,2), residual: true, outChannelsReductionFactor: 2)
    }

    /// Denormalize latents: out = latent * std_of_means + mean_of_means (per-channel).
    /// Input: [B, 128, F, H, W] f16. Stats are broadcast over spatial dims.
    public func denormalize(_ latent: Tensor) -> Tensor {
        let B = latent.shape[0], C = latent.shape[1]
        let F = latent.shape[2], H = latent.shape[3], W = latent.shape[4]
        let spatial = F * H * W
        let out = Tensor.empty(latent.shape, dtype: .float16)
        let pipe = KernelCache.shared.pipeline("channel_scale_bias")
        var cU = UInt32(C), sU = UInt32(spatial)
        MetalContext.shared.run { enc in
            enc.setComputePipelineState(pipe)
            enc.setBuffer(latent.buffer, offset: 0, index: 0)
            enc.setBuffer(stats.stdOfMeans.buffer, offset: 0, index: 1)
            enc.setBuffer(stats.meanOfMeans.buffer, offset: 0, index: 2)
            enc.setBuffer(out.buffer, offset: 0, index: 3)
            enc.setBytes(&cU, length: 4, index: 4)
            enc.setBytes(&sU, length: 4, index: 5)
            enc.dispatchThreads(MTLSize(width: B * C * spatial, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        }
        return out
    }

    /// Denormalize latents f32: out = latent * std + mean (per-channel). All f32.
    public func denormalize(_ buf: MTLBuffer, B: Int, C: Int, F: Int, H: Int, W: Int) -> MTLBuffer {
        stats.denormalize(buf, B: B, C: C, F: F, H: H, W: W)
    }

    /// Normalize latents f32: out = (latent - mean) / std (per-channel). All f32.
    public func normalize(_ buf: MTLBuffer, B: Int, C: Int, F: Int, H: Int, W: Int) -> MTLBuffer {
        stats.normalize(buf, B: B, C: C, F: F, H: H, W: W)
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
            tLast = now
            let line = String(format: "[PERF] +%.1fms %@", dt * 1000, msg)
            log?(line)
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

        // Process up_blocks — load weights on demand
        for (blockIdx, desc) in blockDescs.enumerated() {
            let weights: Any
            switch desc {
            case .midBlock(let idx, let ch, let teDim, let numRes):
                weights = VAEDecoder.loadMidBlock(file: file, prefix: prefix, idx: idx, ch: ch, teDim: teDim, numRes: numRes)
            case .depthToSpace(let idx, let C_in, let C_out):
                weights = VAEDecoder.loadDTSBlock(file: file, prefix: prefix, idx: idx, C_in: C_in, C_out: C_out)
            }

            switch desc {
            case .midBlock(let idx, let ch, let teDim, let numRes):
                step("up_block \(idx): MidBlock(\(C), \(numRes) res), spatial \(D)x\(H)x\(W)\(memTag())")
                let midBlock = weights as! UNetMidBlockWeights
                x = runMidBlock(x, weights: midBlock, B: B, C: C, D: D, H: H, W: W,
                              timestep: tsTensor, scaledTimestep: scaledTimestep, perf: { log?("[PERF] " + $0) })
                pool.reset(keeping: [x.buffer]); pool.purge()
                step("up_block \(idx) done\(memTag())")
                if debug { step(tensorStats(x, label: "up_block_\(idx)_out")) }

            case .depthToSpace(let idx, _, let C_out):
                step("up_block \(idx): DepthToSpace(\(C)→\(C_out)), spatial \(D)x\(H)x\(W)\(memTag())")
                let dtsBlock = weights as! DepthToSpaceWeights
                x = runDepthToSpaceUpsample(x, weights: dtsBlock, B: B, C: C, D: D, H: H, W: W, causal: false, perf: { log?("[PERF] " + $0) })
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
                step("up_block \(idx) done\(memTag())")
                if debug { step(tensorStats(x, label: "up_block_\(idx)_out")) }
            }
        }

        // Tail ops: pixnorm → adaln → silu, tiled over D to avoid 686MB output spike
        step("pixel_norm C=\(C), spatial \(D)x\(H)x\(W)")
        pool.reset(keeping: [x.buffer, tsBuf]); pool.purge()
        let tailBytesPerFrame = B * C * H * W * 2
        let tailTile = max(1, min(D, vaeTailTileBytes / max(tailBytesPerFrame, 1)))
        let tailOutBuf = MetalContext.shared.device.makeBuffer(length: B * C * D * H * W * 2, options: .storageModeShared)!
        let tailOut = Tensor(buffer: tailOutBuf, shape: [B, C, D, H, W], dtype: .float16)
        var td = 0
        while td < D {
            let tc = min(tailTile, D - td)
            var s = temporalRangeGPU(x, B: B, C: C, D: D, H: H, W: W, from: td, count: tc)
            s = pixelNorm3d(s, B: B, C: C, spatialSize: tc * H * W)
            s = applyLastAdaLN(s, B: B, C: C, spatialSize: tc * H * W,
                               timestep: tsTensor, scaledTimestep: scaledTimestep)
            s = silu(s)
            temporalWriteGPU(s, into: tailOut, B: B, C: C, D_out: D, H: H, W: W, dOffset: td, count: tc)
            pool.reset(keeping: [x.buffer, tsBuf, tailOutBuf]); pool.purge()
            td += tc
        }
        pool.reset(keeping: [tailOutBuf, tsBuf]); pool.purge()
        x = tailOut
        if debug {
            step(tensorStats(x, label: "after_silu"))
        }
        step("last AdaLN + silu done\(memTag())")

        // conv_out
        step("conv_out: \(C)→48, 3x3x3, spatial \(D)x\(H)x\(W)")
        pool.reset(keeping: [x.buffer]); pool.purge()
        x = causalConv3d(x, conv: convOut, B: B, C_in: C, D: D, H: H, W: W, causal: false)
        pool.reset(keeping: [x.buffer]); pool.purge()
        if debug { step(tensorStats(x, label: "after_conv_out")) }

        // unpatchify: [B, 48, F, H, W] → [B, 3, F, H*4, W*4]
        step("unpatchify\(memTag())")
        x = unpatchify3d(x, B: B, C_out: 3, F: D, H_in: H, W_in: W, patchSize: patchSize)
        pool.reset(keeping: [x.buffer]); pool.purge()
        if debug { step(tensorStats(x, label: "final_output")) }

        log?("decode complete, shape \(x.shape)\(memTag())")
        return x
    }

    // MARK: - Block execution

    private func causalConv3d(_ x: Tensor, conv: Conv3DWeight, B: Int,
                             C_in: Int, D: Int, H: Int, W: Int, causal: Bool,
                             into: MTLBuffer? = nil) -> Tensor {
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
        return conv3d_implicit(padded, weight: conv.weight, bias: conv.bias,
                               B: B, C_in: C_in, D: D_padded, H: H, W: W,
                               C_out: conv.C_out, kD: conv.kD, kH: conv.kH, kW: conv.kW,
                               sD: conv.sD, sH: conv.sH, sW: conv.sW,
                               pD: 0, pH: spatialPad, pW: spatialPad, into: into)
    }

    /// Pad input for causal/non-causal conv3d, returning the padded tensor.
    /// Call this separately from conv3d so the caller can free the original input.
    private func causalPadForConv(_ x: Tensor, conv: Conv3DWeight, B: Int,
                                  C_in: Int, D: Int, H: Int, W: Int, causal: Bool) -> Tensor {
        var padded = x
        var D_padded = D
        if causal {
            let timePad = conv.kD - 1
            if timePad > 0 {
                padded = causalPad(x, B: B, C: C_in, D: D, H: H, W: W, pad: timePad)
                D_padded = D + timePad
            }
        } else {
            let halfPad = (conv.kD - 1) / 2
            if halfPad > 0 {
                padded = causalPad(x, B: B, C: C_in, D: D, H: H, W: W, pad: halfPad)
                D_padded = D + halfPad
                padded = causalPadBack(padded, B: B, C: C_in, D: D_padded, H: H, W: W, pad: halfPad)
                D_padded = D_padded + halfPad
            }
        }
        return padded
    }

    // causalPadBack is now a GPU kernel in Ops.swift — no CPU memcpy needed

    private func runMidBlock(_ x: Tensor, weights: UNetMidBlockWeights, B: Int, C: Int, D: Int, H: Int, W: Int,
                            timestep: Tensor, scaledTimestep: Float,
                            perf: ((String) -> Void)? = nil) -> Tensor {
        let tsEmb = computeTimestepEmbedding(timestep, embedder: weights.timeEmbedder, B: B,
                                             scaledTimestep: scaledTimestep)

        let pool = MetalContext.shared.bufferPool

        // Choose D-tile size based on spatial footprint to cap peak memory.
        // Each resblock needs ~3× the full-D activation live simultaneously.
        // Target: keep live activations under ~400MB.
        // activation bytes = B * C * D * H * W * 2
        let actBytes = B * C * D * H * W * 2
        let dTile: Int
        // Always tile — conv1OutBuf (full D) + x (full D) are unavoidable, keep slices tiny
        let bytesPerFrame = B * C * H * W * 2
        dTile = max(1, min(D, vaeResTileBytes / max(bytesPerFrame, 1)))

        // Scratch buffer sized for one D-tile (tiled path) or full D (untiled path)
        let scratchD = dTile < D ? dTile : D
        let scratch = MetalContext.shared.device.makeBuffer(
            length: B * C * scratchD * H * W * 2, options: .storageModeShared)!

        let blockStart = CFAbsoluteTimeGetCurrent()
        var h = x
        for (i, resBlock) in weights.resBlocks.enumerated() {
            let t0 = CFAbsoluteTimeGetCurrent()
            h = runResBlock(h, weights: resBlock, B: B, C: C, D: D, H: H, W: W,
                           tsEmb: tsEmb, scratch: scratch, dTile: dTile, perf: perf)
            pool.reset(keeping: [h.buffer, tsEmb.buffer, scratch])
            pool.purge()
            let dt = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            let wall = (CFAbsoluteTimeGetCurrent() - blockStart) * 1000
            perf?(String(format: "  res%d: %.1fms [wall=%.1fms]", i, dt, wall) + memTag())
        }
        return h
    }

    private func runResBlock(_ x: Tensor, weights: ResBlockWeights, B: Int, C: Int, D: Int, H: Int, W: Int,
                            tsEmb: Tensor, scratch: MTLBuffer? = nil, dTile: Int = Int.max,
                            perf: ((String) -> Void)? = nil) -> Tensor {
        let pool = MetalContext.shared.bufferPool

        // AdaLN scale/shift — computed once for full tensor (cheap, no spatial dep on D)
        let (scale1, shift1, scale2, shift2) = computeAdaLN(tsEmb, table: weights.scaleShiftTable, B: B, C: C)

        // If no tiling needed, run the original full-D path
        if dTile >= D {
            return runResBlockFull(x, weights: weights, B: B, C: C, D: D, H: H, W: W,
                                   tsEmb: tsEmb, scale1: scale1, shift1: shift1,
                                   scale2: scale2, shift2: shift2, scratch: scratch)
        }

        // ── Tiled path ──────────────────────────────────────────────────────
        // Key insight: norm (pixel_norm + AdaLN + silu) is per-pixel-channel —
        // it works identically on any D-slice. So we tile EVERYTHING:
        // for each tile: extract x-slice → norm → pad → conv → write to out.
        // Peak live buffers per tile: x (full, read-only) + sliceNorm + slicePadded + sliceOut
        // = D*full + dTile*slices ≈ x_full + 3×dTile_slice. Much better.
        //
        // Conv2 needs conv1's full output for norm2 — so we do a two-pass approach:
        // Pass 1: x → norm1 → conv1 → conv1Out (full D, but written slice by slice)
        // Pass 2: conv1Out → norm2 → conv2 → residual add (slice by slice, no full buffer needed)

        let convOutBytes = B * C * D * H * W * 2
        let conv1OutBuf = MetalContext.shared.device.makeBuffer(length: convOutBytes, options: .storageModeShared)!
        let conv1Out = Tensor(buffer: conv1OutBuf, shape: [B, C, D, H, W], dtype: .float16)

        let spatialPad1 = weights.conv1.kH / 2
        if vaeDebugLogs { perf?("[RES-P1] C=\(C) D=\(D) dTile=\(dTile)") }
        var d = 0
        while d < D {
            let tileCount = min(dTile, D - d)
            let isLastTile1 = (d + tileCount >= D)
            let srcFrom = max(0, d - 1)
            let srcTo   = min(D, d + tileCount + 1)
            let srcCount = srcTo - srcFrom
            let xSlice1 = temporalRangeGPU(x, B: B, C: C, D: D, H: H, W: W, from: srcFrom, count: srcCount)
            var normSlice = pixelNormScaleShiftSilu(xSlice1, scale: scale1, shift: shift1, B: B, C: C, spatialSize: srcCount * H * W)
            pool.reset(keeping: [x.buffer, conv1OutBuf, scale2.buffer, shift2.buffer, tsEmb.buffer, normSlice.buffer]); pool.purge()
            var sliceD = srcCount
            if srcFrom == 0  { normSlice = causalPad(normSlice, B: B, C: C, D: sliceD, H: H, W: W, pad: 1); sliceD += 1 }
            if isLastTile1   { normSlice = causalPadBack(normSlice, B: B, C: C, D: sliceD, H: H, W: W, pad: 1); sliceD += 1 }
            let sliceOut = conv3d_implicit(normSlice, weight: weights.conv1.weight, bias: weights.conv1.bias,
                                           B: B, C_in: C, D: sliceD, H: H, W: W, C_out: C,
                                           kD: weights.conv1.kD, kH: weights.conv1.kH, kW: weights.conv1.kW,
                                           pH: spatialPad1, pW: spatialPad1)
            let convDout1 = (sliceD - weights.conv1.kD) / weights.conv1.sD + 1
            if vaeDebugLogs { perf?("  [RES-P1] d=\(d) srcFrom=\(srcFrom) srcCount=\(srcCount) sliceD=\(sliceD) convDout1=\(convDout1) -> dOffset=\(d)") }
            temporalWriteGPU(sliceOut, into: conv1Out, B: B, C: C, D_out: D, H: H, W: W, dOffset: d, count: convDout1)
            pool.reset(keeping: [x.buffer, conv1OutBuf, scale2.buffer, shift2.buffer, tsEmb.buffer]); pool.purge()
            d += tileCount
        }

        // Pass 2: write output back into x.buffer (residual read before write — safe)
        let outTensor = Tensor(buffer: x.buffer, shape: [B, C, D, H, W], dtype: .float16)
        let spatialPad2 = weights.conv2.kH / 2
        if vaeDebugLogs { perf?("[RES-P2] C=\(C) D=\(D) dTile=\(dTile)") }
        d = 0
        while d < D {
            let tileCount = min(dTile, D - d)
            let isLastTile2 = (d + tileCount >= D)
            let srcFrom = max(0, d - 1)
            let srcTo   = min(D, d + tileCount + 1)
            let srcCount = srcTo - srcFrom
            let c1Slice = temporalRangeGPU(conv1Out, B: B, C: C, D: D, H: H, W: W, from: srcFrom, count: srcCount)
            var norm2Slice = pixelNormScaleShiftSilu(c1Slice, scale: scale2, shift: shift2, B: B, C: C, spatialSize: srcCount * H * W)
            pool.reset(keeping: [x.buffer, conv1OutBuf, tsEmb.buffer, norm2Slice.buffer]); pool.purge()
            var sliceD = srcCount
            if srcFrom == 0  { norm2Slice = causalPad(norm2Slice, B: B, C: C, D: sliceD, H: H, W: W, pad: 1); sliceD += 1 }
            if isLastTile2   { norm2Slice = causalPadBack(norm2Slice, B: B, C: C, D: sliceD, H: H, W: W, pad: 1); sliceD += 1 }
            let conv2Slice = conv3d_implicit(norm2Slice, weight: weights.conv2.weight, bias: weights.conv2.bias,
                                             B: B, C_in: C, D: sliceD, H: H, W: W, C_out: C,
                                             kD: weights.conv2.kD, kH: weights.conv2.kH, kW: weights.conv2.kW,
                                             pH: spatialPad2, pW: spatialPad2)
            let convDout2 = (sliceD - weights.conv2.kD) / weights.conv2.sD + 1
            if vaeDebugLogs { perf?("  [RES-P2] d=\(d) srcFrom=\(srcFrom) srcCount=\(srcCount) sliceD=\(sliceD) convDout2=\(convDout2) -> dOffset=\(d)") }
            // Read residual from x BEFORE writing into x.buffer via outTensor
            let xSlice = temporalRangeGPU(x, B: B, C: C, D: D, H: H, W: W, from: d, count: convDout2)
            let resSlice = elemAdd(xSlice, conv2Slice)
            pool.reset(keeping: [x.buffer, conv1OutBuf, tsEmb.buffer, resSlice.buffer]); pool.purge()
            temporalWriteGPU(resSlice, into: outTensor, B: B, C: C, D_out: D, H: H, W: W, dOffset: d, count: convDout2)
            pool.reset(keeping: [x.buffer, conv1OutBuf, tsEmb.buffer]); pool.purge()
            d += tileCount
        }

        return outTensor
    }

    /// Full-D resblock (no tiling) — original path.
    private func runResBlockFull(_ x: Tensor, weights: ResBlockWeights, B: Int, C: Int, D: Int, H: Int, W: Int,
                                 tsEmb: Tensor, scale1: Tensor, shift1: Tensor,
                                 scale2: Tensor, shift2: Tensor, scratch: MTLBuffer?) -> Tensor {
        let pool = MetalContext.shared.bufferPool
        let spatialSize = D * H * W
        let convOutBytes = B * C * D * H * W * 2
        let scratchBuf = scratch ?? MetalContext.shared.device.makeBuffer(length: convOutBytes, options: .storageModeShared)!
        var h = x

        h = pixelNormScaleShiftSilu(h, scale: scale1, shift: shift1, B: B, C: C, spatialSize: spatialSize)
        var padded = causalPadForConv(h, conv: weights.conv1, B: B, C_in: C, D: D, H: H, W: W, causal: false)
        pool.reset(keeping: [x.buffer, padded.buffer, scratchBuf, scale2.buffer, shift2.buffer, tsEmb.buffer]); pool.purge()
        let spatialPad1 = weights.conv1.kH / 2
        h = conv3d_implicit(padded, weight: weights.conv1.weight, bias: weights.conv1.bias,
                            B: B, C_in: C, D: padded.shape[2], H: H, W: W,
                            C_out: C, kD: weights.conv1.kD, kH: weights.conv1.kH, kW: weights.conv1.kW,
                            pH: spatialPad1, pW: spatialPad1, into: scratchBuf)
        pool.reset(keeping: [x.buffer, h.buffer, scale2.buffer, shift2.buffer, tsEmb.buffer]); pool.purge()
        h = pixelNormScaleShiftSilu(h, scale: scale2, shift: shift2, B: B, C: C, spatialSize: spatialSize)
        padded = causalPadForConv(h, conv: weights.conv2, B: B, C_in: C, D: D, H: H, W: W, causal: false)
        pool.reset(keeping: [x.buffer, padded.buffer, scratchBuf, tsEmb.buffer]); pool.purge()
        let spatialPad2 = weights.conv2.kH / 2
        h = conv3d_implicit(padded, weight: weights.conv2.weight, bias: weights.conv2.bias,
                            B: B, C_in: C, D: padded.shape[2], H: H, W: W,
                            C_out: C, kD: weights.conv2.kD, kH: weights.conv2.kH, kW: weights.conv2.kW,
                            pH: spatialPad2, pW: spatialPad2, into: scratchBuf)
        pool.reset(keeping: [x.buffer, h.buffer, tsEmb.buffer]); pool.purge()
        h = elemAdd(x, h)
        return h
    }

    private func runDepthToSpaceUpsample(_ x: Tensor, weights: DepthToSpaceWeights, B: Int,
                                        C: Int, D: Int, H: Int, W: Int, causal: Bool,
                                        perf: ((String) -> Void)? = nil) -> Tensor {
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

            // Conv path + skip path fused per tile to avoid full-D buffers.
            // Peak per tile: x (full, read-only) + xSlice + convSlice + shuffled + skipSlice + outSlice
            let convOutC = weights.conv.C_out
            let spatialPadDTS = weights.conv.kH / 2
            let shuffleC = convOutC / prodStride  // = skipC * numRepeat
            let outDfull = p1 == 2 ? newD - 1 : newD
            let outBytes = B * shuffleC * outDfull * newH * newW * 2
            let outBuf = MetalContext.shared.device.makeBuffer(length: outBytes, options: .storageModeShared)!
            let outTensor = Tensor(buffer: outBuf, shape: [B, shuffleC, outDfull, newH, newW], dtype: .float16)

            // Choose DTS tile size: target ~150MB per conv slice
            let dtsBytesPerFrame = B * C * H * W * 2
            let dtsTile = max(1, min(D, vaeDtsTileBytes / max(dtsBytesPerFrame * convOutC / C, 1)))

            var t0 = CFAbsoluteTimeGetCurrent()
            if vaeDebugLogs { perf?("[DTS] C=\(C) D=\(D) p1=\(p1) dtsTile=\(dtsTile) outDfull=\(outDfull) shuffleC=\(shuffleC) skipC=\(skipC) numRepeat=\(numRepeat)") }
            var dOff = 0  // output frame offset (in shuffled+sliced space)
            var d = 0
            while d < D {
                let tileCount = min(dtsTile, D - d)
                let isLastTile = (d + tileCount >= D)

                // ── Conv branch ──────────────────────────────────────────
                let srcFrom = max(0, d - 1)
                let srcTo   = min(D, d + tileCount + 1)
                let srcCount = srcTo - srcFrom
                var xSlice = temporalRangeGPU(x, B: B, C: C, D: D, H: H, W: W, from: srcFrom, count: srcCount)
                var sliceD = srcCount
                if srcFrom == 0 { xSlice = causalPad(xSlice, B: B, C: C, D: sliceD, H: H, W: W, pad: 1); sliceD += 1 }
                if isLastTile   { xSlice = causalPadBack(xSlice, B: B, C: C, D: sliceD, H: H, W: W, pad: 1); sliceD += 1 }
                let convSlice = conv3d_implicit(xSlice, weight: weights.conv.weight, bias: weights.conv.bias,
                                               B: B, C_in: C, D: sliceD, H: H, W: W, C_out: convOutC,
                                               kD: weights.conv.kD, kH: weights.conv.kH, kW: weights.conv.kW,
                                               pH: spatialPadDTS, pW: spatialPadDTS)
                let convDout = (sliceD - weights.conv.kD) / weights.conv.sD + 1
                pool.reset(keeping: [x.buffer, outBuf, convSlice.buffer]); pool.purge()
                var shuffled = pixelShuffle3d(convSlice, B: B, C_out: shuffleC, D_in: convDout, H_in: H, W_in: W,
                                             p1: p1, p2: p2, p3: p3)
                pool.reset(keeping: [x.buffer, outBuf, shuffled.buffer]); pool.purge()
                // convDout may be > tileCount due to halo; clamp to tileCount * p1 output frames.
                // The extra frames come from the left halo (frame d-1) that was included for conv correctness
                // but whose output belongs to the previous tile.
                let maxOutFrames = tileCount * p1
                let dropFront = convDout * p1 - maxOutFrames  // frames to drop from front of shuffled (halo output)
                var outFrames = maxOutFrames
                if vaeDebugLogs { perf?("  [DTS-CONV] d=\(d) srcFrom=\(srcFrom) srcCount=\(srcCount) sliceD=\(sliceD) convDout=\(convDout) dropFront=\(dropFront) outFrames=\(outFrames) dOff=\(dOff)") }
                if dropFront > 0 || (p1 == 2 && d == 0) {
                    let dropTotal = dropFront + (p1 == 2 && d == 0 ? 1 : 0)
                    let tmp2 = shuffled
                    shuffled = temporalSlice(tmp2, B: B, C: shuffleC, D: convDout * p1, H: newH, W: newW, from: dropTotal)
                    if p1 == 2 && d == 0 { outFrames -= 1 }
                    pool.reset(keeping: [x.buffer, outBuf, shuffled.buffer]); pool.purge()
                    if vaeDebugLogs { perf?("  [DTS-CONV] dropped \(dropTotal) frames from front, outFrames now=\(outFrames)") }
                }

                // ── Skip branch for this tile ────────────────────────────
                // Non-tiled reference: skip = pixelShuffle(x)[1..] (drop frame 0, same as conv).
                // Output position k uses skip frame k+1, which comes from input frame (k+1)/p1.
                let skipInFrom  = (dOff + 1) / p1
                let skipInLast  = (dOff + outFrames) / p1
                let skipInCount = skipInLast - skipInFrom + 1
                let skipShuffledD = skipInCount * p1
                let skipFrameOffset = (dOff + 1) - skipInFrom * p1
                if vaeDebugLogs { perf?("  [DTS-SKIP] d=\(d) dOff=\(dOff) outFrames=\(outFrames) | skipInFrom=\(skipInFrom) skipInCount=\(skipInCount) skipShuffledD=\(skipShuffledD) skipFrameOffset=\(skipFrameOffset)") }
                var skipSlice = temporalRangeGPU(x, B: B, C: C, D: D, H: H, W: W, from: skipInFrom, count: skipInCount)
                pool.reset(keeping: [x.buffer, outBuf, shuffled.buffer, skipSlice.buffer]); pool.purge()
                var skipShuffled = pixelShuffle3d(skipSlice, B: B, C_out: skipC, D_in: skipInCount, H_in: H, W_in: W,
                                                 p1: p1, p2: p2, p3: p3)
                if numRepeat > 1 {
                    let tmp = skipShuffled
                    skipShuffled = repeatChannels(tmp, B: B, C: skipC, spatialSize: skipInCount * p1 * newH * newW, repeats: numRepeat)
                }
                if skipFrameOffset > 0 || skipShuffledD > outFrames {
                    let tmp = skipShuffled
                    skipShuffled = temporalRangeGPU(tmp, B: B, C: shuffleC, D: skipShuffledD, H: newH, W: newW,
                                                    from: skipFrameOffset, count: outFrames)
                }
                pool.reset(keeping: [x.buffer, outBuf, shuffled.buffer, skipShuffled.buffer]); pool.purge()

                // ── Add + write ──────────────────────────────────────────
                let added = elemAdd(shuffled, skipShuffled)
                pool.reset(keeping: [x.buffer, outBuf, added.buffer]); pool.purge()
                temporalWriteGPU(added, into: outTensor, B: B, C: shuffleC, D_out: outDfull, H: newH, W: newW,
                                 dOffset: dOff, count: outFrames)
                pool.reset(keeping: [x.buffer, outBuf]); pool.purge()
                dOff += outFrames
                d += tileCount
            }
            var h = outTensor
            perf?(String(format: "  dts_conv(%d→%d): %.1fms", C, convOutC, (CFAbsoluteTimeGetCurrent() - t0) * 1000) + memTag())
            perf?(String(format: "  dts_shuffle: 0.0ms (fused into tiled conv)"))
            pool.reset(keeping: [h.buffer]); pool.purge()

            result = h
        } else {
            var t0nb = CFAbsoluteTimeGetCurrent()
            var h = causalConv3d(x, conv: weights.conv, B: B, C_in: C, D: D, H: H, W: W, causal: causal)
            perf?(String(format: "  dts_conv(%d→%d): %.1fms", C, weights.conv.C_out, (CFAbsoluteTimeGetCurrent() - t0nb) * 1000) + memTag())
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
