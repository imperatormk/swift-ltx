import Foundation
import Metal
import SwiftLLM

private func debugF32Stats(_ label: String, _ buf: MTLBuffer, count: Int) -> String {
    let ptr = buf.contents().assumingMemoryBound(to: Float.self)
    var mn: Float = .infinity, mx: Float = -.infinity, sum: Float = 0, nanCount = 0, infCount = 0
    for i in 0..<count {
        let v = ptr[i]
        if v.isNaN { nanCount += 1; continue }
        if v.isInfinite { infCount += 1 }
        if v < mn { mn = v }
        if v > mx { mx = v }
        sum += v
    }
    return String(format: "[DEBUG] %@: min=%.4g max=%.4g mean=%.4g nan=%d inf=%d (n=%d)",
                  label, mn, mx, sum / Float(count - nanCount), nanCount, infCount, count)
}

@MainActor
final class VideoRunner: ObservableObject {
    @Published var isRunning = false
    @Published var statusText = "Ready"
    @Published var debugLog = ""
    @Published var decodedImage: DecodedFrame?  // legacy single frame
    @Published var decodedFrames: [DecodedFrame] = []
    @Published var progress: (Int, Int)? = nil

    #if os(macOS)
    @Published var ditPath = NSString(string: "~/Downloads/ltxv_dit_bf16.safetensors").expandingTildeInPath
    @Published var embeddingsPath = NSString(string: "~/Downloads/t5_embed_a_cat_walking_on_a_sunny_beach.safetensors").expandingTildeInPath
    @Published var vaePath = NSString(string: "~/Downloads/ltxv_vae_decoder_f16.safetensors").expandingTildeInPath
    @Published var upsamplerPath = NSString(string: "~/Downloads/ltxv-spatial-upscaler-0.9.8.safetensors").expandingTildeInPath
    @Published var t5WeightsPath = NSString(string: "~/Downloads/t5_v1_1_xxl_encoder_4bit_v2.safetensors").expandingTildeInPath
    @Published var t5TokenizerPath = NSString(string: "~/.cache/huggingface/hub/models--google-t5--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1/tokenizer.json").expandingTildeInPath
    #else
    @Published var ditPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0] + "/ltxv_dit_bf16.safetensors"
    @Published var embeddingsPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0] + "/t5_embed_a_cat_walking_on_a_sunny_beach.safetensors"
    @Published var vaePath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0] + "/ltxv_vae_decoder_f16.safetensors"
    @Published var upsamplerPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0] + "/ltxv-spatial-upscaler-0.9.8.safetensors"
    @Published var t5WeightsPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0] + "/t5_v1_1_xxl_encoder_4bit_v2.safetensors"
    @Published var t5TokenizerPath = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)[0] + "/tokenizer.json"
    #endif

    @Published var useT5Encoder = true
    @Published var customPrompt = "young woman"

    @Published var peakMemText: String = "—"

    @Published var numFrames: Int = 2
    @Published var height: Int = 24
    @Published var width: Int = 24
    @Published var seed: UInt64 = 42

    var deviceName: String { MetalContext.shared.device.name }

    /// Load a fresh DiT pipeline (caller is responsible for letting it go out of scope to free memory).
    /// Not MainActor-isolated — safe to call from background threads.
    nonisolated private func loadDiTPipeline(ditPath: String, log: ((String) -> Void)? = nil) throws -> LTXPipeline {
        let start = CFAbsoluteTimeGetCurrent()
        let ditURL = URL(fileURLWithPath: ditPath)
        let ditWeights = try loadDiTWeights(from: ditURL)
        let ditModel = DiTModel(weights: ditWeights)
        let pipeline = LTXPipeline(model: ditModel)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        log?(String(format: "DiT loaded in %.1fs (%d blocks)", elapsed, ditWeights.config.numLayers))
        return pipeline
    }

    /// Load VAE decoder on demand. Pass pre-loaded stats to avoid re-loading them.
    nonisolated private func loadVAEDecoder(vaePath: String, stats: VAELatentStats? = nil, log: ((String) -> Void)? = nil) throws -> VAEDecoder {
        let start = CFAbsoluteTimeGetCurrent()
        let vaeURL = URL(fileURLWithPath: vaePath)
        let vaeFile = try SafeTensorsFile(url: vaeURL)
        let prefix: String
        if vaeFile.tensors.keys.contains(where: { $0.hasPrefix("decoder.") }) {
            prefix = "decoder."
        } else if vaeFile.tensors.keys.contains(where: { $0.hasPrefix("vae.decoder.") }) {
            prefix = "vae.decoder."
        } else {
            throw VideoRunnerError.noDecoderKeys
        }
        let vae = try VAEDecoder(file: vaeFile, prefix: prefix, stats: stats)
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        log?(String(format: "VAE loaded in %.1fs", elapsed))
        return vae
    }

    func generate() {
        guard !isRunning else { return }

        let embPath = embeddingsPath
        guard !embPath.isEmpty else { statusText = "Set embeddings path"; return }

        isRunning = true
        statusText = "Generating..."
        debugLog = ""
        decodedImage = nil
        decodedFrames = []
        progress = nil

        let nFrames = numFrames
        let h = height
        let w = width
        let s = seed
        let upPath = upsamplerPath
        let ditPth = ditPath
        let vaePth = vaePath
        let useT5 = useT5Encoder
        let t5Weights = t5WeightsPath
        let t5Tok = t5TokenizerPath
        let prompt = customPrompt

        // Distilled model timesteps (from LTX-Video config)
        let pass1Timesteps: [Float] = [1.0, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250]
        let pass2Timesteps: [Float] = [0.9094, 0.7250, 0.4219]

        Thread.detachNewThread { [weak self] in
            guard let self else { return }
            do {
                var log = ""
                let start = CFAbsoluteTimeGetCurrent()

                func updateLog(_ msg: String, mem: Bool = false) {
                    guard msg.contains("[PERF]") else { return }
                    log += msg + "\n"
                    let snap = log
                    let (_, peak) = PeakMemoryTracker.shared.sample()
                    let peakStr = String(format: "%dMB", peak / 1048576)
                    DispatchQueue.main.async { [weak self] in
                        self?.debugLog = snap
                        self?.peakMemText = peakStr
                    }
                }

                func updateStatus(_ msg: String) {
                    DispatchQueue.main.async { [weak self] in self?.statusText = msg }
                }

                // Load T5 embeddings (either from file or by running T5 encoder)
                var pt = CFAbsoluteTimeGetCurrent()
                let textEmb: Tensor
                let textSeqLen: Int

                if useT5 {
                    // Run T5-XXL 4-bit encoder (scoped to free weights after)
                    let (t5Buf, t5Len, t5Time): (MTLBuffer, Int, Double) = try autoreleasepool {
                        updateStatus("Loading T5 encoder...")
                        let tokenizer = try T5Tokenizer(url: URL(fileURLWithPath: t5Tok))
                        let tokenIds = tokenizer.encode(prompt)
                        updateLog(String(format: "[PERF] t5_tokenize: %d tokens", tokenIds.count))

                        updateStatus("Running T5 encoder (24 layers)...")
                        let encoder = try T5Encoder(url: URL(fileURLWithPath: t5Weights))
                        encoder.log = { msg in updateLog("[PERF] \(msg)") }
                        let maxLen = 128
                        let embBuf = encoder.encode(tokenIds: tokenIds, maxLength: maxLen)

                        let realTokens = min(tokenIds.count, maxLen)
                        let dim = 4096
                        let bytes = realTokens * dim * 4
                        let buf = MetalContext.shared.device.makeBuffer(length: bytes, options: .storageModeShared)!
                        memcpy(buf.contents(), embBuf.contents(), bytes)
                        let elapsed = (CFAbsoluteTimeGetCurrent() - pt) * 1000
                        return (buf, realTokens, elapsed)
                        // encoder + tokenizer freed here
                    }
                    MetalContext.shared.bufferPool.releaseAll(keeping: [t5Buf])
                    textSeqLen = t5Len
                    textEmb = Tensor(buffer: t5Buf, shape: [1, t5Len, 4096], dtype: .float32)
                    updateLog(String(format: "[PERF] t5_encode: %.1fms (seqLen=%d), Metal=%dMB", t5Time, textSeqLen, MetalContext.shared.device.currentAllocatedSize/1048576))
                } else {
                    // Load pre-computed embeddings
                    let embURL = URL(fileURLWithPath: embPath)
                    let (textEmbFull, textMask) = try loadTextEmbeddings(from: embURL)
                    let fullSeqLen = textEmbFull.shape[1]
                    let realTokens: Int
                    if let textMask {
                        var count = 0
                        if textMask.dtype == .float32 {
                            let maskPtr = textMask.buffer.contents().assumingMemoryBound(to: Float.self)
                            for i in 0..<(textMask.count) { if maskPtr[i] > 0.5 { count += 1 } }
                        } else {
                            let maskPtr = textMask.buffer.contents().assumingMemoryBound(to: Float16.self)
                            for i in 0..<(textMask.count) { if Float(maskPtr[i]) > 0.5 { count += 1 } }
                        }
                        realTokens = count
                    } else {
                        realTokens = fullSeqLen
                    }
                    textSeqLen = realTokens
                    if realTokens < fullSeqLen {
                        let dim = textEmbFull.shape[2]
                        let bytes = realTokens * dim * 4
                        let buf = MetalContext.shared.device.makeBuffer(length: bytes, options: .storageModeShared)!
                        memcpy(buf.contents(), textEmbFull.buffer.contents(), bytes)
                        textEmb = Tensor(buffer: buf, shape: [1, realTokens, dim], dtype: .float32)
                    } else {
                        textEmb = textEmbFull
                    }
                    updateLog(String(format: "[PERF] t5_embeddings_load: %.1fms (seqLen=%d)", (CFAbsoluteTimeGetCurrent() - pt) * 1000, textSeqLen))
                }

                // Load VAE latent stats once (~1KB), then close the file to free mmap
                pt = CFAbsoluteTimeGetCurrent()
                let latentStats: VAELatentStats = try autoreleasepool {
                    let vaeFile = try SafeTensorsFile(url: URL(fileURLWithPath: vaePth))
                    let vaePrefix: String
                    if vaeFile.tensors.keys.contains(where: { $0.hasPrefix("decoder.") }) {
                        vaePrefix = "decoder."
                    } else if vaeFile.tensors.keys.contains(where: { $0.hasPrefix("vae.decoder.") }) {
                        vaePrefix = "vae.decoder."
                    } else {
                        throw VideoRunnerError.noDecoderKeys
                    }
                    return VAELatentStats.load(file: vaeFile, prefix: vaePrefix)
                    // vaeFile (mmap) freed here
                }
                updateLog(String(format: "[PERF] vae_stats_load: %.1fms", (CFAbsoluteTimeGetCurrent() - pt) * 1000))

                // Two-pass: pass1 at ~2/3 target, upsampler 2x, pass2 at full
                let vaeScaleFactor = 32  // 8x compression × 4 patch
                let downscale: Float = 0.6667
                let targetPixH = h * vaeScaleFactor
                let targetPixW = w * vaeScaleFactor
                let dsH = Int(Float(targetPixH) * downscale)
                let dsW = Int(Float(targetPixW) * downscale)
                let firstPixH = dsH - (dsH % vaeScaleFactor)
                let firstPixW = dsW - (dsW % vaeScaleFactor)
                let pass1H = firstPixH / vaeScaleFactor
                let pass1W = firstPixW / vaeScaleFactor
                let finalH = pass1H * 2
                let finalW = pass1W * 2

                // === Stage 1: Load DiT → Pass 1 → Release DiT ===
                let latentC = 128  // config.outChannels
                let pass1Count = 1 * latentC * nFrames * pass1H * pass1W
                let pass1F32: MTLBuffer = autoreleasepool {
                    pt = CFAbsoluteTimeGetCurrent()
                    updateStatus("Loading DiT for Pass 1...")
                    var pipeline: LTXPipeline? = try! self.loadDiTPipeline(ditPath: ditPth) { msg in updateLog(msg) }
                    updateLog(String(format: "[PERF] S1 dit_load: %.1fms", (CFAbsoluteTimeGetCurrent() - pt) * 1000))

                    updateStatus("Pass 1 denoising...")
                    pt = CFAbsoluteTimeGetCurrent()
                    let result = pipeline!.generateLatents(
                        textEmbeddings: textEmb,
                        textSeqLen: textSeqLen,
                        numFrames: nFrames,
                        height: pass1H,
                        width: pass1W,
                        explicitTimesteps: pass1Timesteps,
                        seed: s,
                        progressHandler: { step, total in
                            let snap = "Pass 1 step \(step)/\(total)"
                            DispatchQueue.main.async { [weak self] in
                                self?.progress = (step, total)
                                self?.statusText = snap
                            }
                        },
                        log: { msg in updateLog("  [PERF] P1 \(msg)") }
                    )
                    updateLog(String(format: "[PERF] S1 pass1_denoise: %.1fms (%d steps)", (CFAbsoluteTimeGetCurrent() - pt) * 1000, pass1Timesteps.count))
                    pipeline = nil
                    return result
                }
                let dev = MetalContext.shared.device
                MetalContext.shared.bufferPool.releaseAll(keeping: [pass1F32])

                // === Stage 2: Upsampler + AdaIN (no VAE needed — stats already loaded) ===
                // updateLog(debugF32Stats("pass1F32_input", pass1F32, count: pass1Count))
                let upCount = 1 * latentC * nFrames * finalH * finalW
                let adainedF32: MTLBuffer = autoreleasepool {
                    var s2t = CFAbsoluteTimeGetCurrent()
                    updateStatus("Loading upsampler...")
                    let loader = try! UpsamplerWeightLoader(url: URL(fileURLWithPath: upPath))
                    updateLog(String(format: "[PERF] S2 upsampler_open: %.1fms, Metal=%dMB", (CFAbsoluteTimeGetCurrent() - s2t) * 1000, dev.currentAllocatedSize/1048576), mem: true)

                    updateStatus("Upsampling latents (f32)...")
                    s2t = CFAbsoluteTimeGetCurrent()
                    let denormedPass1F32 = latentStats.denormalize(pass1F32, B: 1, C: latentC, F: nFrames, H: pass1H, W: pass1W)
                    updateLog(String(format: "[PERF] S2 denormalize: %.1fms", (CFAbsoluteTimeGetCurrent() - s2t) * 1000), mem: true)

                    s2t = CFAbsoluteTimeGetCurrent()
                    let upsampledF32 = upsamplerForward(denormedPass1F32, loader: loader, B: 1, F: nFrames, H: pass1H, W: pass1W) { msg in
                        updateLog("  \(msg)", mem: msg.contains("[PERF]"))
                    }
                    updateLog(String(format: "[PERF] S2 upsampler_forward: %.1fms", (CFAbsoluteTimeGetCurrent() - s2t) * 1000), mem: true)

                    s2t = CFAbsoluteTimeGetCurrent()
                    let renormedF32 = latentStats.normalize(upsampledF32, B: 1, C: latentC, F: nFrames, H: finalH, W: finalW)
                    updateLog(String(format: "[PERF] S2 normalize: %.1fms", (CFAbsoluteTimeGetCurrent() - s2t) * 1000), mem: true)

                    s2t = CFAbsoluteTimeGetCurrent()
                    let result = adainFilter(renormedF32, reference: pass1F32,
                                             B: 1, C: latentC,
                                             latShape: [1, latentC, nFrames, finalH, finalW],
                                             refShape: [1, latentC, nFrames, pass1H, pass1W],
                                             factor: 1.0)
                    updateLog(String(format: "[PERF] S2 adain: %.1fms", (CFAbsoluteTimeGetCurrent() - s2t) * 1000), mem: true)
                    return result
                    // loader freed here
                }
                let s2 = MetalContext.shared.bufferPool.stats
                updateLog(String(format: "S2 end: Metal=%dMB pool(act=%dMB/%d free=%dMB/%d)",
                    dev.currentAllocatedSize/1048576, s2.active/1048576, s2.activeCount, s2.free/1048576, s2.freeCount), mem: true)
                MetalContext.shared.bufferPool.releaseAll(keeping: [adainedF32])

                // === Stage 3: Load DiT → Pass 2 → Release DiT ===
                pt = CFAbsoluteTimeGetCurrent()
                let pass2NoiseOverride: MTLBuffer? = nil  // Generate fresh noise (MLX noise has wrong shape for Pass 2)
                updateLog(String(format: "[PERF] S3 noise_load: %.1fms", (CFAbsoluteTimeGetCurrent() - pt) * 1000))

                let pass2Count = 1 * latentC * nFrames * finalH * finalW
                let pass2F32: MTLBuffer = autoreleasepool {
                    pt = CFAbsoluteTimeGetCurrent()
                    updateStatus("Loading DiT for Pass 2...")
                    var pipeline2: LTXPipeline? = try! self.loadDiTPipeline(ditPath: ditPth) { msg in updateLog(msg) }
                    updateLog(String(format: "[PERF] S3 dit_load: %.1fms", (CFAbsoluteTimeGetCurrent() - pt) * 1000))

                    updateStatus("Pass 2 denoising...")
                    pt = CFAbsoluteTimeGetCurrent()
                    let result = pipeline2!.denoiseLatents(
                        inputLatentsF32: adainedF32,
                        B: 1, C: latentC, numFrames: nFrames, height: finalH, width: finalW,
                        textEmbeddings: textEmb,
                        textSeqLen: textSeqLen,
                        timesteps: pass2Timesteps,
                        seed: s,
                        noiseOverrideF32: pass2NoiseOverride,
                        progressHandler: { step, total in
                            let snap = "Pass 2 step \(step)/\(total)"
                            DispatchQueue.main.async { [weak self] in
                                self?.progress = (step, total)
                                self?.statusText = snap
                            }
                        },
                        log: { msg in updateLog("  [PERF] P2 \(msg)") }
                    )
                    updateLog(String(format: "[PERF] S3 pass2_denoise: %.1fms (%d steps)", (CFAbsoluteTimeGetCurrent() - pt) * 1000, pass2Timesteps.count))
                    pipeline2 = nil
                    return result
                }
                MetalContext.shared.bufferPool.releaseAll(keeping: [pass2F32])

                // === Stage 4: Load VAE → decode ===
                pt = CFAbsoluteTimeGetCurrent()
                updateStatus("Loading VAE for decode...")
                let vaeDecoder = try self.loadVAEDecoder(vaePath: vaePth, stats: latentStats) { msg in updateLog(msg) }
                updateLog(String(format: "[PERF] S4 vae_load: %.1fms", (CFAbsoluteTimeGetCurrent() - pt) * 1000))

                pt = CFAbsoluteTimeGetCurrent()
                let finalF16 = castF32toF16(pass2F32, count: pass2Count, shape: [1, latentC, nFrames, finalH, finalW])
                updateLog(String(format: "[PERF] S4 cast_f32_to_f16: %.1fms", (CFAbsoluteTimeGetCurrent() - pt) * 1000))

                pt = CFAbsoluteTimeGetCurrent()
                let denormed = vaeDecoder.denormalize(finalF16)
                updateLog(String(format: "[PERF] S4 denormalize: %.1fms", (CFAbsoluteTimeGetCurrent() - pt) * 1000))

                pt = CFAbsoluteTimeGetCurrent()
                let noised = mixDecodeNoise(denormed, noiseScale: 0.025, seed: 0)
                updateLog(String(format: "[PERF] S4 decode_noise: %.1fms", (CFAbsoluteTimeGetCurrent() - pt) * 1000))

                updateStatus("VAE decoding...")
                pt = CFAbsoluteTimeGetCurrent()
                let output = vaeDecoder.decode(noised) { msg in
                    updateLog("  [PERF] VAE \(msg)")
                }
                updateLog(String(format: "[PERF] S4 vae_decode: %.1fms", (CFAbsoluteTimeGetCurrent() - pt) * 1000))

                guard output.shape.count == 5 else {
                    updateLog("ERROR: expected 5D output, got \(output.shape)")
                    updateStatus("Error: VAE output shape \(output.shape)")
                    DispatchQueue.main.async { [weak self] in
                        self?.isRunning = false
                        self?.progress = nil
                    }
                    return
                }

                let totalElems = output.shape.reduce(1, *)
                let bufElems = output.buffer.length / 2
                guard bufElems >= totalElems else {
                    updateLog("ERROR: buffer too small \(bufElems) < \(totalElems)")
                    updateStatus("Error: buffer too small")
                    DispatchQueue.main.async { [weak self] in
                        self?.isRunning = false
                        self?.progress = nil
                    }
                    return
                }

                let frames = extractAllFrames(from: output, targetWidth: targetPixW, targetHeight: targetPixH)
                updateLog("\(frames.count) frames: \(frames.first?.width ?? 0)x\(frames.first?.height ?? 0)")

                let totalTime = CFAbsoluteTimeGetCurrent() - start
                updateLog(String(format: "[PERF] TOTAL: %.1fms (%.1fs)", totalTime * 1000, totalTime))

                let finalLog = log
                DispatchQueue.main.async { [weak self] in
                    self?.decodedFrames = frames
                    self?.decodedImage = frames.first
                    self?.debugLog = finalLog
                    self?.statusText = String(format: "Done in %.1fs — %d frames %dx%d", totalTime, frames.count, frames.first?.width ?? 0, frames.first?.height ?? 0)
                    self?.isRunning = false
                    self?.progress = nil
                }
            } catch {
                DispatchQueue.main.async { [weak self] in
                    self?.statusText = "Error: \(error)"
                    self?.isRunning = false
                    self?.progress = nil
                }
            }
        }
    }

    func generateFromMLXAdaIN() {
        guard !isRunning else { return }

        isRunning = true
        statusText = "Loading MLX AdaIN latents..."
        debugLog = ""
        decodedImage = nil

        let embPath = embeddingsPath
        let ditPth = ditPath
        let vaePth = vaePath
        let pass2Timesteps: [Float] = [0.9094, 0.7250, 0.4219]

        Thread.detachNewThread { [weak self] in
            guard let self else { return }
            do {
                var log = ""
                func updateLog(_ msg: String) {
                    log += msg + "\n"; let s = log
                    let (_, peak) = PeakMemoryTracker.shared.sample()
                    let peakStr = String(format: "%dMB", peak / 1048576)
                    DispatchQueue.main.async { [weak self] in self?.debugLog = s; self?.peakMemText = peakStr }
                }
                func updateStatus(_ msg: String) { DispatchQueue.main.async { [weak self] in self?.statusText = msg } }

                // Load MLX AdaIN latents
                let adainFile = try SafeTensorsFile(url: URL(fileURLWithPath: "/tmp/mlx_adain_latents.safetensors"))
                let key = adainFile.tensors.keys.first!
                let info = adainFile.tensors[key]!
                let ptr = adainFile.pointer(for: key)!
                updateLog("MLX AdaIN: \(key) \(info.shape) \(info.dtype)")

                // Convert to f32 MTLBuffer
                let count = info.shape.reduce(1, *)
                let adainF32: MTLBuffer
                if info.dtype == .float32 {
                    adainF32 = MetalContext.shared.device.makeBuffer(bytes: ptr, length: count * 4, options: .storageModeShared)!
                } else {
                    let src = ptr.assumingMemoryBound(to: Float16.self)
                    adainF32 = MetalContext.shared.device.makeBuffer(length: count * 4, options: .storageModeShared)!
                    let dst = adainF32.contents().assumingMemoryBound(to: Float.self)
                    for i in 0..<count { dst[i] = Float(src[i]) }
                }
                // updateLog("AdaIN f32: \(f32Stats(adainF32, count: count))")

                let B = info.shape[0], latentC = info.shape[1], nFrames = info.shape[2]
                let finalH = info.shape[3], finalW = info.shape[4]

                // Load text embeddings
                let embURL = URL(fileURLWithPath: embPath)
                let (textEmbFull, textMask) = try loadTextEmbeddings(from: embURL)
                let fullSeqLen = textEmbFull.shape[1]
                let realTokens: Int
                if let textMask {
                    var c = 0
                    if textMask.dtype == .float32 {
                        let mp = textMask.buffer.contents().assumingMemoryBound(to: Float.self)
                        for i in 0..<textMask.count { if mp[i] > 0.5 { c += 1 } }
                    } else {
                        let mp = textMask.buffer.contents().assumingMemoryBound(to: Float16.self)
                        for i in 0..<textMask.count { if Float(mp[i]) > 0.5 { c += 1 } }
                    }
                    realTokens = c
                } else { realTokens = fullSeqLen }
                let textEmb: Tensor
                if realTokens < fullSeqLen {
                    let dim = textEmbFull.shape[2]
                    let bytes = realTokens * dim * 4
                    let buf = MetalContext.shared.device.makeBuffer(length: bytes, options: .storageModeShared)!
                    memcpy(buf.contents(), textEmbFull.buffer.contents(), bytes)
                    textEmb = Tensor(buffer: buf, shape: [1, realTokens, dim], dtype: .float32)
                } else { textEmb = textEmbFull }

                // Pass 2 — load DiT on demand
                updateStatus("Loading DiT for Pass 2...")
                let pipeline = try self.loadDiTPipeline(ditPath: ditPth) { msg in updateLog(msg) }
                updateStatus("Pass 2 denoising...")
                let pass2F32 = pipeline.denoiseLatents(
                    inputLatentsF32: adainF32,
                    B: B, C: latentC, numFrames: nFrames, height: finalH, width: finalW,
                    textEmbeddings: textEmb, textSeqLen: realTokens,
                    timesteps: pass2Timesteps, seed: 42,
                    progressHandler: { step, total in
                        DispatchQueue.main.async { [weak self] in self?.progress = (step, total); self?.statusText = "Pass 2 step \(step)/\(total)" }
                    },
                    log: { msg in updateLog("  DiT: \(msg)") }
                )
                let pass2Count = B * latentC * nFrames * finalH * finalW
                // updateLog("Pass 2 latents f32: \(f32Stats(pass2F32, count: pass2Count))")

                // VAE decode — load on demand
                updateStatus("Loading VAE...")
                let vaeDecoder = try self.loadVAEDecoder(vaePath: vaePth) { msg in updateLog(msg) }
                let finalF16 = castF32toF16(pass2F32, count: pass2Count, shape: [B, latentC, nFrames, finalH, finalW])
                let denormed = vaeDecoder.denormalize(finalF16)
                let noised = mixDecodeNoise(denormed, noiseScale: 0.025, seed: 0)
                updateStatus("VAE decoding...")
                let output = vaeDecoder.decode(noised) { msg in updateLog("  VAE: \(msg)") }
                guard output.shape.count == 5 else {
                    updateStatus("Error: VAE output shape \(output.shape)")
                    DispatchQueue.main.async { [weak self] in self?.isRunning = false; self?.progress = nil }
                    return
                }
                let frame = extractFirstFrame(from: output)
                updateLog("Frame: \(frame.width)x\(frame.height)")
                let finalLog = log
                DispatchQueue.main.async { [weak self] in
                    self?.decodedImage = frame; self?.debugLog = finalLog
                    self?.statusText = "Done (MLX AdaIN) — \(frame.width)x\(frame.height)"
                    self?.isRunning = false; self?.progress = nil
                }
            } catch {
                DispatchQueue.main.async { [weak self] in self?.statusText = "Error: \(error)"; self?.isRunning = false; self?.progress = nil }
            }
        }
    }
}

enum VideoRunnerError: Error, CustomStringConvertible {
    case noDecoderKeys

    var description: String {
        switch self {
        case .noDecoderKeys: return "No decoder keys found in VAE safetensors"
        }
    }
}

// Uses extractFirstFrame() and DecodedFrame from VAERunner.swift
