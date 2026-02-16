import Foundation
import Metal
import SwiftLLM

@MainActor
final class VideoRunner: ObservableObject {
    @Published var isRunning = false
    @Published var statusText = "Ready"
    @Published var debugLog = ""
    @Published var decodedImage: DecodedFrame?
    @Published var progress: (Int, Int)? = nil

    #if os(macOS)
    @Published var ditPath = NSString(string: "~/Downloads/ltxv_dit_bf16.safetensors").expandingTildeInPath
    @Published var embeddingsPath = NSString(string: "~/Downloads/t5_embed_a_cat_walking_on_a_sunny_beach.safetensors").expandingTildeInPath
    @Published var vaePath = NSString(string: "~/Downloads/ltxv_vae_decoder_f16.safetensors").expandingTildeInPath
    @Published var upsamplerPath = NSString(string: "~/Downloads/ltxv-spatial-upscaler-0.9.8.safetensors").expandingTildeInPath
    #else
    @Published var ditPath = ""
    @Published var embeddingsPath = ""
    @Published var vaePath = ""
    @Published var upsamplerPath = ""
    #endif

    @Published var numFrames: Int = 2
    @Published var height: Int = 24
    @Published var width: Int = 24
    @Published var seed: UInt64 = 42

    private var vaeDecoder: VAEDecoder?

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

    func loadModels() {
        guard !isRunning else { return }
        isRunning = true
        statusText = "Loading VAE..."
        debugLog = ""

        let vaePth = vaePath

        Thread.detachNewThread { [weak self] in
            guard let self else { return }
            do {
                let start = CFAbsoluteTimeGetCurrent()

                // Load VAE only (small, kept resident)
                let vaeURL = URL(fileURLWithPath: vaePth)
                let vaeFile = try SafeTensorsFile(url: vaeURL)
                let prefix: String
                if vaeFile.tensors.keys.contains(where: { $0.hasPrefix("decoder.") }) {
                    prefix = "decoder."
                } else if vaeFile.tensors.keys.contains(where: { $0.hasPrefix("vae.decoder.") }) {
                    prefix = "vae.decoder."
                } else {
                    throw VideoRunnerError.noDecoderKeys
                }
                let vaeDecoder = try VAEDecoder(file: vaeFile, prefix: prefix)
                let vaeTime = CFAbsoluteTimeGetCurrent() - start

                let msg = String(format: "VAE loaded in %.1fs\n", vaeTime)
                DispatchQueue.main.async { [weak self] in
                    self?.vaeDecoder = vaeDecoder
                    self?.debugLog += msg
                    self?.statusText = String(format: "VAE loaded (%.1fs). DiT loaded on demand per stage.", vaeTime)
                    self?.isRunning = false
                }
            } catch {
                DispatchQueue.main.async { [weak self] in
                    self?.statusText = "Error: \(error)"
                    self?.isRunning = false
                }
            }
        }
    }

    func generate() {
        guard !isRunning else { return }
        guard let vaeDecoder else { statusText = "Load VAE first"; return }

        let embPath = embeddingsPath
        guard !embPath.isEmpty else { statusText = "Set embeddings path"; return }

        isRunning = true
        statusText = "Generating..."
        debugLog = ""
        decodedImage = nil
        progress = nil

        let nFrames = numFrames
        let h = height
        let w = width
        let s = seed
        let upPath = upsamplerPath
        let ditPth = ditPath

        // Distilled model timesteps (from LTX-Video config)
        let pass1Timesteps: [Float] = [1.0, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250]
        let pass2Timesteps: [Float] = [0.9094, 0.7250, 0.4219]

        Thread.detachNewThread { [weak self] in
            guard let self else { return }
            do {
                var log = ""
                let start = CFAbsoluteTimeGetCurrent()

                func updateLog(_ msg: String) {
                    log += msg + "\n"
                    let snap = log
                    DispatchQueue.main.async { [weak self] in self?.debugLog = snap }
                }

                func updateStatus(_ msg: String) {
                    DispatchQueue.main.async { [weak self] in self?.statusText = msg }
                }

                // Load T5 embeddings
                let embURL = URL(fileURLWithPath: embPath)
                let (textEmbFull, textMask) = try loadTextEmbeddings(from: embURL)
                // Truncate to real tokens using attention mask (skip padding)
                let fullSeqLen = textEmbFull.shape[1]
                let realTokens: Int
                if let textMask {
                    // Count real tokens (mask values > 0.5) — mask may be f32 or f16
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
                let textSeqLen = realTokens
                let textEmb: Tensor
                if realTokens < fullSeqLen {
                    // Truncate: copy first realTokens from [1, fullSeqLen, 4096] → [1, realTokens, 4096]
                    let dim = textEmbFull.shape[2]
                    let bytes = realTokens * dim * 4  // f32
                    let buf = MetalContext.shared.device.makeBuffer(length: bytes, options: .storageModeShared)!
                    memcpy(buf.contents(), textEmbFull.buffer.contents(), bytes)
                    textEmb = Tensor(buffer: buf, shape: [1, realTokens, dim], dtype: .float32)
                    updateLog("T5 embeddings: \(textEmbFull.shape) → truncated to \(textEmb.shape) (\(realTokens) real tokens)")
                } else {
                    textEmb = textEmbFull
                    updateLog("T5 embeddings: \(textEmb.shape) seqLen=\(textSeqLen)")
                }

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
                let pass1F32: MTLBuffer
                do {
                    updateStatus("Loading DiT for Pass 1...")
                    let pipeline = try self.loadDiTPipeline(ditPath: ditPth) { msg in updateLog(msg) }

                    updateStatus("Pass 1 denoising...")
                    updateLog("=== Pass 1: \(nFrames)f × \(pass1H)×\(pass1W) ===")
                    let ditStart = CFAbsoluteTimeGetCurrent()
                    pass1F32 = pipeline.generateLatents(
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
                        log: { msg in updateLog("  DiT: \(msg)") }
                    )
                    let ditTime = CFAbsoluteTimeGetCurrent() - ditStart
                    updateLog(String(format: "Pass 1 done: %.2fs (%d steps)", ditTime, pass1Timesteps.count))
                    updateLog("Pass 1 latents f32: \(f32Stats(pass1F32, count: pass1Count))")
                    // pipeline goes out of scope here → ARC releases DiT weights
                }
                MetalContext.shared.bufferPool.releaseAll(keeping: [pass1F32])
                updateLog("Released DiT weights + all intermediates after Pass 1")

                // DEBUG: skip upsampler+pass2, decode pass1 directly
                let skipPass2 = false
                if skipPass2 {
                    let pass1F16 = castF32toF16(pass1F32, count: pass1Count, shape: [1, latentC, nFrames, pass1H, pass1W])
                    let denormed = vaeDecoder.denormalize(pass1F16)
                    updateLog("Denormalized pass1: \(f16Stats(denormed))")
                    let noised = mixDecodeNoise(denormed, noiseScale: 0.025, seed: 0)
                    updateStatus("VAE decoding (pass1 only)...")
                    let output = vaeDecoder.decode(noised) { msg in updateLog("  VAE: \(msg)") }
                    updateLog("Output: \(output.shape)")
                    guard output.shape.count == 5 else {
                        updateStatus("Error: VAE output shape \(output.shape)")
                        DispatchQueue.main.async { [weak self] in self?.isRunning = false; self?.progress = nil }
                        return
                    }
                    let frame = extractFirstFrame(from: output)
                    updateLog("Frame: \(frame.width)x\(frame.height)")
                    let totalTime = CFAbsoluteTimeGetCurrent() - start
                    let finalLog = log
                    DispatchQueue.main.async { [weak self] in
                        self?.decodedImage = frame
                        self?.debugLog = finalLog
                        self?.statusText = String(format: "Done (pass1 only) %.1fs — %dx%d", totalTime, frame.width, frame.height)
                        self?.isRunning = false; self?.progress = nil
                    }
                    return
                }

                // === Stage 2: Upsample + AdaIN (all f32) ===
                let upCount = 1 * latentC * nFrames * finalH * finalW
                let adainedF32: MTLBuffer
                do {
                    updateStatus("Loading upsampler...")
                    let upStart = CFAbsoluteTimeGetCurrent()
                    let upWeights = try loadUpsamplerWeights(from: URL(fileURLWithPath: upPath))
                    let upsampler = LatentUpsampler(weights: upWeights)
                    updateLog(String(format: "Upsampler loaded in %.1fs", CFAbsoluteTimeGetCurrent() - upStart))

                    updateStatus("Upsampling latents (f32)...")
                    let denormedPass1F32 = vaeDecoder.denormalize(pass1F32, B: 1, C: latentC, F: nFrames, H: pass1H, W: pass1W)
                    updateLog("Denormalized pass1 f32: \(f32Stats(denormedPass1F32, count: pass1Count))")

                    let upsampledF32 = upsampler.forward(denormedPass1F32, B: 1, F: nFrames, H: pass1H, W: pass1W) { msg in
                        updateLog("  Upsampler: \(msg)")
                    }
                    updateLog("Upsampled f32: \(f32Stats(upsampledF32, count: upCount))")

                    let renormedF32 = vaeDecoder.normalize(upsampledF32, B: 1, C: latentC, F: nFrames, H: finalH, W: finalW)
                    updateLog("Re-normalized f32: \(f32Stats(renormedF32, count: upCount))")

                    adainedF32 = adainFilter(renormedF32, reference: pass1F32,
                                                     B: 1, C: latentC,
                                                     latShape: [1, latentC, nFrames, finalH, finalW],
                                                     refShape: [1, latentC, nFrames, pass1H, pass1W],
                                                     factor: 1.0)
                    updateLog("AdaIN f32: \(f32Stats(adainedF32, count: upCount))")
                    // upWeights, upsampler, denormedPass1F32, upsampledF32, renormedF32
                    // all go out of scope here → ARC releases upsampler weights + intermediates
                }
                MetalContext.shared.bufferPool.releaseAll(keeping: [adainedF32])
                updateLog("Released upsampler + all intermediates after Stage 2")

                // === Stage 3: Load DiT → Pass 2 → Release DiT ===
                // Load MLX pass2 noise if available
                var pass2NoiseOverride: MTLBuffer? = nil
                let pass2NoiseURL = URL(fileURLWithPath: "/tmp/mlx_pass2_noise.safetensors")
                if FileManager.default.fileExists(atPath: pass2NoiseURL.path) {
                    let nf = try SafeTensorsFile(url: pass2NoiseURL)
                    let nk = nf.tensors.keys.first!
                    let ni = nf.tensors[nk]!
                    let np = nf.pointer(for: nk)!
                    let nc = ni.shape.reduce(1, *)
                    if ni.dtype == .float32 {
                        pass2NoiseOverride = MetalContext.shared.device.makeBuffer(bytes: np, length: nc * 4, options: .storageModeShared)!
                    } else {
                        let src = np.assumingMemoryBound(to: Float16.self)
                        pass2NoiseOverride = MetalContext.shared.device.makeBuffer(length: nc * 4, options: .storageModeShared)!
                        let dst = pass2NoiseOverride!.contents().assumingMemoryBound(to: Float.self)
                        for i in 0..<nc { dst[i] = Float(src[i]) }
                    }
                    updateLog("Loaded MLX pass2 noise: \(f32Stats(pass2NoiseOverride!, count: nc))")
                }

                let pass2Count = 1 * latentC * nFrames * finalH * finalW
                let pass2F32: MTLBuffer
                do {
                    updateStatus("Loading DiT for Pass 2...")
                    let pipeline = try self.loadDiTPipeline(ditPath: ditPth) { msg in updateLog(msg) }

                    updateStatus("Pass 2 denoising...")
                    updateLog("=== Pass 2: \(nFrames)f × \(finalH)×\(finalW) ===")
                    let pass2Start = CFAbsoluteTimeGetCurrent()
                    pass2F32 = pipeline.denoiseLatents(
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
                        log: { msg in updateLog("  DiT: \(msg)") }
                    )
                    let pass2Time = CFAbsoluteTimeGetCurrent() - pass2Start
                    updateLog(String(format: "Pass 2 done: %.2fs (%d steps)", pass2Time, pass2Timesteps.count))
                    updateLog("Pass 2 latents f32: \(f32Stats(pass2F32, count: pass2Count))")
                    // pipeline goes out of scope here → ARC releases DiT weights
                }
                MetalContext.shared.bufferPool.releaseAll(keeping: [pass2F32])
                updateLog("Released DiT weights after Pass 2")

                // === VAE decode (cast f32→f16 once) ===
                updateLog("Casting to f16 for VAE...")
                let finalF16 = castF32toF16(pass2F32, count: pass2Count, shape: [1, latentC, nFrames, finalH, finalW])
                let denormed = vaeDecoder.denormalize(finalF16)
                updateLog("Denormalized: \(f16Stats(denormed))")

                let noised = mixDecodeNoise(denormed, noiseScale: 0.025, seed: 0)
                updateLog("After decode noise: \(f16Stats(noised))")

                updateStatus("VAE decoding...")
                let vaeStart = CFAbsoluteTimeGetCurrent()
                let output = vaeDecoder.decode(noised) { msg in
                    updateLog("  VAE: \(msg)")
                }
                let vaeTime = CFAbsoluteTimeGetCurrent() - vaeStart
                updateLog(String(format: "VAE decode: %.2fs", vaeTime))
                updateLog("Output: \(output.shape)")

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

                let frame = extractFirstFrame(from: output)
                updateLog("Frame: \(frame.width)x\(frame.height)")

                let totalTime = CFAbsoluteTimeGetCurrent() - start
                updateLog(String(format: "Total: %.2fs", totalTime))

                let finalLog = log
                DispatchQueue.main.async { [weak self] in
                    self?.decodedImage = frame
                    self?.debugLog = finalLog
                    self?.statusText = String(format: "Done in %.1fs — %dx%d", totalTime, frame.width, frame.height)
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
        guard let vaeDecoder else { statusText = "Load VAE first"; return }

        isRunning = true
        statusText = "Loading MLX AdaIN latents..."
        debugLog = ""
        decodedImage = nil

        let embPath = embeddingsPath
        let ditPth = ditPath
        let pass2Timesteps: [Float] = [0.9094, 0.7250, 0.4219]

        Thread.detachNewThread { [weak self] in
            guard let self else { return }
            do {
                var log = ""
                func updateLog(_ msg: String) { log += msg + "\n"; let s = log; DispatchQueue.main.async { [weak self] in self?.debugLog = s } }
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
                updateLog("AdaIN f32: \(f32Stats(adainF32, count: count))")

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

                // Load MLX pass2 noise if available
                var noiseOverride: MTLBuffer? = nil
                let noiseURL = URL(fileURLWithPath: "/tmp/mlx_pass2_noise.safetensors")
                if FileManager.default.fileExists(atPath: noiseURL.path) {
                    let noiseFile = try SafeTensorsFile(url: noiseURL)
                    let noiseKey = noiseFile.tensors.keys.first!
                    let noiseInfo = noiseFile.tensors[noiseKey]!
                    let noisePtr = noiseFile.pointer(for: noiseKey)!
                    let noiseCount = noiseInfo.shape.reduce(1, *)
                    if noiseInfo.dtype == .float32 {
                        noiseOverride = MetalContext.shared.device.makeBuffer(bytes: noisePtr, length: noiseCount * 4, options: .storageModeShared)!
                    } else {
                        let src = noisePtr.assumingMemoryBound(to: Float16.self)
                        noiseOverride = MetalContext.shared.device.makeBuffer(length: noiseCount * 4, options: .storageModeShared)!
                        let dst = noiseOverride!.contents().assumingMemoryBound(to: Float.self)
                        for i in 0..<noiseCount { dst[i] = Float(src[i]) }
                    }
                    updateLog("Loaded MLX pass2 noise: \(noiseInfo.shape) → \(f32Stats(noiseOverride!, count: noiseCount))")
                } else {
                    updateLog("No MLX noise file, using srand48 noise")
                }

                // Pass 2 — load DiT on demand
                updateStatus("Loading DiT for Pass 2...")
                let pipeline = try self.loadDiTPipeline(ditPath: ditPth) { msg in updateLog(msg) }
                updateStatus("Pass 2 denoising (MLX input)...")
                let pass2F32 = pipeline.denoiseLatents(
                    inputLatentsF32: adainF32,
                    B: B, C: latentC, numFrames: nFrames, height: finalH, width: finalW,
                    textEmbeddings: textEmb, textSeqLen: realTokens,
                    timesteps: pass2Timesteps, seed: 42,
                    noiseOverrideF32: noiseOverride,
                    progressHandler: { step, total in
                        DispatchQueue.main.async { [weak self] in self?.progress = (step, total); self?.statusText = "Pass 2 step \(step)/\(total)" }
                    },
                    log: { msg in updateLog("  DiT: \(msg)") }
                )
                let pass2Count = B * latentC * nFrames * finalH * finalW
                updateLog("Pass 2 latents f32: \(f32Stats(pass2F32, count: pass2Count))")

                // VAE decode
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
