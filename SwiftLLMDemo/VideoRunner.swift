import Foundation
import Metal
import SwiftLLM

struct VideoGenerationRequest {
    let ditPath: String
    let vaePath: String
    let upsamplerPath: String
    let embeddingsPath: String
    let useT5Encoder: Bool
    let t5WeightsPath: String
    let t5TokenizerPath: String
    let prompt: String
    let numFrames: Int
    let height: Int
    let width: Int
    let seed: UInt64
}

struct VideoGenerationProgress: Equatable {
    let label: String
    let step: Int
    let total: Int
}

struct VideoGenerationResult {
    let frames: [DecodedFrame]
    let log: String
    let peakMemoryText: String
    let summary: String
}

enum VideoGenerationEvent {
    case status(String)
    case progress(VideoGenerationProgress?)
    case log(String)
    case peakMemory(String)
}

enum VideoGenerationError: Error, CustomStringConvertible {
    case missingPath(String)
    case noDecoderKeys
    case cancelled

    var description: String {
        switch self {
        case .missingPath(let label):
            return "Missing path: \(label)"
        case .noDecoderKeys:
            return "No decoder keys found in VAE safetensors"
        case .cancelled:
            return "Generation cancelled"
        }
    }
}

final class VideoGenerationCancellation: @unchecked Sendable {
    private let lock = os_unfair_lock_t.allocate(capacity: 1)
    private var cancelled = false

    init() {
        lock.initialize(to: os_unfair_lock())
    }

    func cancel() {
        os_unfair_lock_lock(lock)
        cancelled = true
        os_unfair_lock_unlock(lock)
    }

    var isCancelled: Bool {
        os_unfair_lock_lock(lock)
        let value = cancelled
        os_unfair_lock_unlock(lock)
        return value
    }

    deinit {
        lock.deinitialize(count: 1)
        lock.deallocate()
    }
}

private struct VideoLogCollector {
    private(set) var text = ""

    mutating func append(_ line: String) {
        text += line + "\n"
    }
}

final class VideoGenerationService: @unchecked Sendable {
    private let pass1Timesteps: [Float] = [1.0, 0.9937, 0.9875, 0.9812, 0.9750, 0.9094, 0.7250]
    private let pass2Timesteps: [Float] = [0.9094, 0.7250, 0.4219]

    var deviceName: String { MetalContext.shared.device.name }

    func generate(
        request: VideoGenerationRequest,
        cancellation: VideoGenerationCancellation,
        emit: @escaping (VideoGenerationEvent) -> Void
    ) throws -> VideoGenerationResult {
        PeakMemoryTracker.shared.reset()
        var log = VideoLogCollector()
        let start = CFAbsoluteTimeGetCurrent()

        func updateStatus(_ status: String) {
            emit(.status(status))
        }

        func updateProgress(_ progress: VideoGenerationProgress?) {
            emit(.progress(progress))
        }

        func updateLog(_ line: String) {
            log.append(line)
            let (_, peak) = PeakMemoryTracker.shared.sample()
            emit(.log(log.text))
            emit(.peakMemory(String(format: "%dMB", peak / 1048576)))
        }

        func checkCancellation() throws {
            if cancellation.isCancelled {
                throw VideoGenerationError.cancelled
            }
        }

        func runStage<T>(_ title: String, body: () throws -> T) throws -> T {
            try checkCancellation()
            updateStatus(title)
            return try body()
        }

        let textEmbeddings = try runStage("Preparing prompt...") {
            try loadTextConditioning(request: request, updateLog: updateLog)
        }
        let latentStats = try loadLatentStats(vaePath: request.vaePath)
        updateLog("[PERF] VAE stats loaded")

        let latentChannels = 128
        let vaeScaleFactor = 32
        let targetPixelHeight = request.height * vaeScaleFactor
        let targetPixelWidth = request.width * vaeScaleFactor
        let downscale: Float = 0.6667
        let firstPixelHeight = alignedLatentPixelSize(target: targetPixelHeight, downscale: downscale, scaleFactor: vaeScaleFactor)
        let firstPixelWidth = alignedLatentPixelSize(target: targetPixelWidth, downscale: downscale, scaleFactor: vaeScaleFactor)
        let pass1Height = firstPixelHeight / vaeScaleFactor
        let pass1Width = firstPixelWidth / vaeScaleFactor
        let finalHeight = pass1Height * 2
        let finalWidth = pass1Width * 2

        let pass1Latents = try runStage("Pass 1: loading DiT...") {
            let pipeline = try loadDiTPipeline(path: request.ditPath, updateLog: updateLog)
            updateStatus("Pass 1: denoising...")
            return try pipeline.generateLatents(
                textEmbeddings: textEmbeddings.tensor,
                textSeqLen: textEmbeddings.sequenceLength,
                numFrames: request.numFrames,
                height: pass1Height,
                width: pass1Width,
                explicitTimesteps: pass1Timesteps,
                seed: request.seed,
                progressHandler: { step, total in
                    updateProgress(VideoGenerationProgress(label: "Pass 1", step: step, total: total))
                },
                log: { message in
                    updateLog("[PERF] P1 \(message)")
                }
            )
        }
        MetalContext.shared.bufferPool.releaseAll(keeping: [pass1Latents])
        try checkCancellation()

        let stage2Start = CFAbsoluteTimeGetCurrent()
        let refinedLatents = try runStage("Pass 2 prep: upsampling latents...") {
            let loader = try UpsamplerWeightLoader(url: URL(fileURLWithPath: request.upsamplerPath))
            let denormalized = latentStats.denormalize(pass1Latents, B: 1, C: latentChannels, F: request.numFrames, H: pass1Height, W: pass1Width)
            let upsampled = try upsamplerForward(
                denormalized,
                loader: loader,
                B: 1,
                F: request.numFrames,
                H: pass1Height,
                W: pass1Width,
                log: { message in
                    updateLog(message)
                }
            )
            let normalized = latentStats.normalize(upsampled, B: 1, C: latentChannels, F: request.numFrames, H: finalHeight, W: finalWidth)
            return adainFilter(
                normalized,
                reference: pass1Latents,
                B: 1,
                C: latentChannels,
                latShape: [1, latentChannels, request.numFrames, finalHeight, finalWidth],
                refShape: [1, latentChannels, request.numFrames, pass1Height, pass1Width],
                factor: 1.0
            )
        }
        updateLog(String(format: "[PERF] Stage 2 total: %.1fms", (CFAbsoluteTimeGetCurrent() - stage2Start) * 1000))
        MetalContext.shared.bufferPool.releaseAll(keeping: [refinedLatents])
        try checkCancellation()

        let pass2Latents = try runStage("Pass 2: loading DiT...") {
            let pipeline = try loadDiTPipeline(path: request.ditPath, updateLog: updateLog)
            updateStatus("Pass 2: denoising...")
            return try pipeline.denoiseLatents(
                inputLatentsF32: refinedLatents,
                B: 1,
                C: latentChannels,
                numFrames: request.numFrames,
                height: finalHeight,
                width: finalWidth,
                textEmbeddings: textEmbeddings.tensor,
                textSeqLen: textEmbeddings.sequenceLength,
                timesteps: pass2Timesteps,
                seed: request.seed,
                progressHandler: { step, total in
                    updateProgress(VideoGenerationProgress(label: "Pass 2", step: step, total: total))
                },
                log: { message in
                    updateLog("[PERF] P2 \(message)")
                }
            )
        }
        MetalContext.shared.bufferPool.releaseAll(keeping: [pass2Latents])
        try checkCancellation()

        let frames = try runStage("Decoding video...") {
            let decoder = try loadVAEDecoder(path: request.vaePath, stats: latentStats)
            let pass2Count = latentChannels * request.numFrames * finalHeight * finalWidth
            let finalF16 = castF32toF16(pass2Latents, count: pass2Count, shape: [1, latentChannels, request.numFrames, finalHeight, finalWidth])
            let denormalized = decoder.denormalize(finalF16)
            let noised = mixDecodeNoise(denormalized, noiseScale: 0.025, seed: 0)
            let decoded = decoder.decode(noised) { message in
                updateLog("[PERF] VAE \(message)")
            }
            return extractAllFrames(from: decoded, targetWidth: targetPixelWidth, targetHeight: targetPixelHeight)
        }

        updateProgress(nil)
        let totalTime = CFAbsoluteTimeGetCurrent() - start
        updateLog(String(format: "[PERF] TOTAL: %.1fms", totalTime * 1000))
        let peakText = String(format: "%dMB", PeakMemoryTracker.shared.peak / 1048576)

        return VideoGenerationResult(
            frames: frames,
            log: log.text,
            peakMemoryText: peakText,
            summary: String(format: "Done in %.1fs, %d frames at %dx%d", totalTime, frames.count, frames.first?.width ?? 0, frames.first?.height ?? 0)
        )
    }

    private func alignedLatentPixelSize(target: Int, downscale: Float, scaleFactor: Int) -> Int {
        let scaled = Int(Float(target) * downscale)
        return scaled - (scaled % scaleFactor)
    }

    private func loadDiTPipeline(path: String, updateLog: (String) -> Void) throws -> LTXPipeline {
        let start = CFAbsoluteTimeGetCurrent()
        let weights = try loadDiTWeights(from: URL(fileURLWithPath: path))
        updateLog(String(format: "[PERF] DiT weights loaded in %.1fms", (CFAbsoluteTimeGetCurrent() - start) * 1000))
        return LTXPipeline(model: DiTModel(weights: weights))
    }

    private func loadVAEDecoder(path: String, stats: VAELatentStats?) throws -> VAEDecoder {
        let file = try SafeTensorsFile(url: URL(fileURLWithPath: path))
        let prefix = try detectVAEPrefix(file: file)
        return try VAEDecoder(file: file, prefix: prefix, stats: stats)
    }

    private func loadLatentStats(vaePath: String) throws -> VAELatentStats {
        let file = try SafeTensorsFile(url: URL(fileURLWithPath: vaePath))
        let prefix = try detectVAEPrefix(file: file)
        return VAELatentStats.load(file: file, prefix: prefix)
    }

    private func detectVAEPrefix(file: SafeTensorsFile) throws -> String {
        if file.tensors.keys.contains(where: { $0.hasPrefix("decoder.") }) {
            return "decoder."
        }
        if file.tensors.keys.contains(where: { $0.hasPrefix("vae.decoder.") }) {
            return "vae.decoder."
        }
        throw VideoGenerationError.noDecoderKeys
    }

    private func loadTextConditioning(
        request: VideoGenerationRequest,
        updateLog: @escaping (String) -> Void
    ) throws -> (tensor: Tensor, sequenceLength: Int) {
        if request.useT5Encoder {
            let tokenizer = try T5Tokenizer(url: URL(fileURLWithPath: request.t5TokenizerPath))
            let tokenIds = tokenizer.encode(request.prompt)
            let maxLength = 128
            let encoder = try T5Encoder(url: URL(fileURLWithPath: request.t5WeightsPath))
            encoder.log = { message in
                updateLog("[PERF] T5 \(message)")
            }
            let encoded = encoder.encode(tokenIds: tokenIds, maxLength: maxLength)
            let sequenceLength = min(tokenIds.count, maxLength)
            let dim = 4096
            let bytes = sequenceLength * dim * 4
            let copy = MetalContext.shared.device.makeBuffer(length: bytes, options: .storageModeShared)!
            memcpy(copy.contents(), encoded.contents(), bytes)
            MetalContext.shared.bufferPool.releaseAll(keeping: [copy])
            updateLog("[PERF] T5 prompt encoding complete")
            return (Tensor(buffer: copy, shape: [1, sequenceLength, dim], dtype: .float32), sequenceLength)
        }

        let loaded = try loadTextEmbeddings(from: URL(fileURLWithPath: request.embeddingsPath))
        let trimmed = trimTextEmbeddings(loaded.0, attentionMask: loaded.1)
        updateLog("[PERF] Precomputed text embeddings loaded")
        return trimmed
    }

    private func trimTextEmbeddings(_ embeddings: Tensor, attentionMask: Tensor?) -> (tensor: Tensor, sequenceLength: Int) {
        let fullSequenceLength = embeddings.shape[1]
        let sequenceLength: Int
        if let attentionMask {
            if attentionMask.dtype == .float32 {
                let values = attentionMask.buffer.contents().assumingMemoryBound(to: Float.self)
                sequenceLength = (0..<attentionMask.count).reduce(into: 0) { count, index in
                    if values[index] > 0.5 { count += 1 }
                }
            } else {
                let values = attentionMask.buffer.contents().assumingMemoryBound(to: Float16.self)
                sequenceLength = (0..<attentionMask.count).reduce(into: 0) { count, index in
                    if Float(values[index]) > 0.5 { count += 1 }
                }
            }
        } else {
            sequenceLength = fullSequenceLength
        }

        guard sequenceLength < fullSequenceLength else {
            return (embeddings, fullSequenceLength)
        }

        let dim = embeddings.shape[2]
        let bytes = sequenceLength * dim * 4
        let trimmed = MetalContext.shared.device.makeBuffer(length: bytes, options: .storageModeShared)!
        memcpy(trimmed.contents(), embeddings.buffer.contents(), bytes)
        return (Tensor(buffer: trimmed, shape: [1, sequenceLength, dim], dtype: .float32), sequenceLength)
    }
}

@MainActor
final class VideoGenerationViewModel: ObservableObject {
    @Published var isRunning = false
    @Published var statusText = "Ready"
    @Published var debugLog = ""
    @Published var decodedFrames: [DecodedFrame] = []
    @Published var progress: VideoGenerationProgress? = nil
    @Published var peakMemText = "—"

    #if os(macOS)
    @Published var ditPath = NSString(string: "~/Models/SwiftLLM/ltxv_dit_bf16.safetensors").expandingTildeInPath
    @Published var embeddingsPath = NSString(string: "~/Models/SwiftLLM/t5_embed_a_cat_walking_on_a_sunny_beach.safetensors").expandingTildeInPath
    @Published var vaePath = NSString(string: "~/Models/SwiftLLM/ltxv_vae_decoder_f16.safetensors").expandingTildeInPath
    @Published var upsamplerPath = NSString(string: "~/Models/SwiftLLM/ltxv-spatial-upscaler-0.9.8.safetensors").expandingTildeInPath
    @Published var t5WeightsPath = NSString(string: "~/Models/SwiftLLM/t5_v1_1_xxl_encoder_4bit_v2.safetensors").expandingTildeInPath
    @Published var t5TokenizerPath = NSString(string: "~/.cache/huggingface/hub/models--google-t5--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1/tokenizer.json").expandingTildeInPath
    #else
    @Published var ditPath = ""
    @Published var embeddingsPath = ""
    @Published var vaePath = ""
    @Published var upsamplerPath = ""
    @Published var t5WeightsPath = ""
    @Published var t5TokenizerPath = ""
    #endif

    @Published var useT5Encoder = true
    @Published var customPrompt = "young woman"
    @Published var numFrames = 8
    @Published var height = 24
    @Published var width = 24
    @Published var seed: UInt64 = 42

    private let service = VideoGenerationService()
    private var cancellation = VideoGenerationCancellation()
    private var generationThread: Thread?

    var deviceName: String { service.deviceName }

    func generate() {
        guard !isRunning else { return }
        do {
            let request = try makeRequest()
            let cancellation = VideoGenerationCancellation()
            let service = self.service
            self.cancellation = cancellation

            isRunning = true
            statusText = "Preparing..."
            debugLog = ""
            decodedFrames = []
            progress = nil
            peakMemText = "—"

            generationThread = Thread { [weak self] in
                guard let self else { return }
                do {
                    let result = try service.generate(request: request, cancellation: cancellation) { event in
                        DispatchQueue.main.async { [weak self] in
                            self?.apply(event: event)
                        }
                    }
                    DispatchQueue.main.async { [weak self] in
                        self?.decodedFrames = result.frames
                        self?.debugLog = result.log
                        self?.peakMemText = result.peakMemoryText
                        self?.statusText = result.summary
                        self?.progress = nil
                        self?.isRunning = false
                        self?.generationThread = nil
                    }
                } catch {
                    let message = error.localizedDescription
                    DispatchQueue.main.async { [weak self] in
                        self?.statusText = "Error: \(message)"
                        self?.progress = nil
                        self?.isRunning = false
                        self?.generationThread = nil
                    }
                }
            }
            generationThread?.qualityOfService = .userInitiated
            generationThread?.start()
        } catch {
            statusText = "Error: \(error.localizedDescription)"
        }
    }

    func cancel() {
        guard isRunning else { return }
        cancellation.cancel()
        statusText = "Cancelling..."
    }

    private func makeRequest() throws -> VideoGenerationRequest {
        if ditPath.isEmpty { throw VideoGenerationError.missingPath("DiT weights") }
        if vaePath.isEmpty { throw VideoGenerationError.missingPath("VAE weights") }
        if upsamplerPath.isEmpty { throw VideoGenerationError.missingPath("Upsampler weights") }
        if useT5Encoder {
            if t5WeightsPath.isEmpty { throw VideoGenerationError.missingPath("T5 weights") }
            if t5TokenizerPath.isEmpty { throw VideoGenerationError.missingPath("T5 tokenizer") }
        } else if embeddingsPath.isEmpty {
            throw VideoGenerationError.missingPath("Text embeddings")
        }

        return VideoGenerationRequest(
            ditPath: ditPath,
            vaePath: vaePath,
            upsamplerPath: upsamplerPath,
            embeddingsPath: embeddingsPath,
            useT5Encoder: useT5Encoder,
            t5WeightsPath: t5WeightsPath,
            t5TokenizerPath: t5TokenizerPath,
            prompt: customPrompt,
            numFrames: numFrames,
            height: height,
            width: width,
            seed: seed
        )
    }

    private func apply(event: VideoGenerationEvent) {
        switch event {
        case .status(let status):
            statusText = status
        case .progress(let progress):
            self.progress = progress
        case .log(let log):
            debugLog = log
        case .peakMemory(let peakMemory):
            peakMemText = peakMemory
        }
    }
}
