import Foundation
import Metal
import SwiftLLM

struct GenerationResult: Identifiable {
    let id = UUID()
    let prompt: String
    let output: String
    let tokenCount: Int
    let elapsed: Double
    let ttft: Double
    let tokPerSec: Double
    let usedFlash: Bool
}

@MainActor
final class DemoRunner: ObservableObject {
    @Published var isRunning = false
    @Published var statusText = "Ready"
    @Published var liveOutput = ""
    @Published var results: [GenerationResult] = []
    @Published var prompt = "Hello, who are you?"
    @Published var useFlashAttention = true
    #if os(macOS)
    @Published var modelPath = "/Users/zimski/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"
    #else
    @Published var modelPath = {
        // Auto-detect model in Documents/model/
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let modelDir = docs.appendingPathComponent("model")
        if FileManager.default.fileExists(atPath: modelDir.appendingPathComponent("config.json").path) {
            return modelDir.path
        }
        return ""
    }()
    #endif
    @Published var modelURL: URL?

    private var model: LlamaModel?
    private var tokenizer: Tokenizer?
    nonisolated(unsafe) private var _shouldStop = false

    func stop() {
        _shouldStop = true
    }

    var deviceName: String {
        MetalContext.shared.device.name
    }

    func run() {
        guard !isRunning else { return }
        _shouldStop = false
        let text = prompt
        let url = modelURL
        let path = url?.path ?? modelPath
        let flash = useFlashAttention
        isRunning = true
        liveOutput = ""
        statusText = "Preparing..."

        let model = self.model
        let tokenizer = self.tokenizer

        Thread.detachNewThread { [weak self] in
            self?.generateOnThread(text: text, path: path, flash: flash,
                                   modelURL: url,
                                   existingModel: model, existingTokenizer: tokenizer)
        }
    }

    private nonisolated func generateOnThread(text: String, path: String, flash: Bool,
                                               modelURL url: URL?,
                                               existingModel: LlamaModel?,
                                               existingTokenizer: Tokenizer?) {
        var model = existingModel
        var tokenizer = existingTokenizer

        if model == nil {
            guard !path.isEmpty else {
                DispatchQueue.main.async { [weak self] in
                    self?.statusText = "Set model path first"
                    self?.isRunning = false
                }
                return
            }
            let dir: URL
            if let url {
                dir = url
                _ = url.startAccessingSecurityScopedResource()
            } else {
                dir = URL(fileURLWithPath: path)
            }
            DispatchQueue.main.async { [weak self] in self?.statusText = "Loading tokenizer..." }
            do {
                tokenizer = try Tokenizer(from: dir.appendingPathComponent("tokenizer.json"))
                DispatchQueue.main.async { [weak self] in self?.statusText = "Loading weights..." }
                model = try LlamaModel(directory: dir)
                let m = model!
                DispatchQueue.main.async { [weak self] in
                    self?.model = m
                    self?.tokenizer = tokenizer
                    self?.statusText = "Model loaded (\(m.config.numHiddenLayers)L, \(m.config.numAttentionHeads)H)"
                }
            } catch {
                DispatchQueue.main.async { [weak self] in
                    self?.statusText = "Error: \(error)"
                    self?.isRunning = false
                }
                return
            }
        }

        guard let model, let tokenizer else { return }
        model.useFlashAttention = flash

        let tokens = tokenizer.chatTokens(for: text)
        DispatchQueue.main.async { [weak self] in
            self?.statusText = "Generating (\(tokens.count) prompt tokens)..."
            self?.liveOutput = ""
        }

        let start = CFAbsoluteTimeGetCurrent()
        var firstTokenTime: CFAbsoluteTime?
        var count = 0
        var fullOutput = ""

        model.generate(prompt: tokens, maxTokens: 256) { tokenId, _ in
            if firstTokenTime == nil {
                firstTokenTime = CFAbsoluteTimeGetCurrent()
            }
            if tokenId == tokenizer.eosToken || tokenId == tokenizer.eotId { return false }
            if self._shouldStop { return false }

            let decoded = tokenizer.decode(tokenId)
            fullOutput += decoded
            count += 1

            let snapshot = fullOutput
            DispatchQueue.main.async { [weak self] in
                self?.liveOutput = snapshot
            }
            return true
        }

        let end = CFAbsoluteTimeGetCurrent()
        let elapsed = end - start
        let ttft = firstTokenTime.map { $0 - start } ?? elapsed
        let decodeTime = firstTokenTime.map { end - $0 } ?? elapsed
        let tps = count > 1 ? Double(count - 1) / decodeTime : (count > 0 ? Double(count) / decodeTime : 0)

        let result = GenerationResult(
            prompt: text, output: fullOutput,
            tokenCount: count, elapsed: elapsed,
            ttft: ttft, tokPerSec: tps, usedFlash: flash)

        DispatchQueue.main.async { [weak self] in
            self?.results.insert(result, at: 0)
            self?.statusText = String(format: "Done: %d tokens, %.1f tok/s, TTFT %.2fs", count, tps, ttft)
            self?.isRunning = false
        }
    }
}
