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
    let prefillTokPerSec: Double
    let usedFlash: Bool
}

@MainActor
final class DemoRunner: ObservableObject {
    @Published var isRunning = false
    @Published var statusText = "Ready"
    @Published var liveOutput = ""
    @Published var debugLog = ""
    @Published var results: [GenerationResult] = []
    @Published var prompt = "Hello, who are you?"
    @Published var useFlashAttention = true
    @Published var useFastGEMM = true
    @Published var temperature: Float = 0.7
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
    private var hasAutoLoaded = false

    func stop() {
        _shouldStop = true
    }

    func autoLoadIfNeeded() {
        guard !hasAutoLoaded, model == nil, !modelPath.isEmpty, !isRunning else { return }
        hasAutoLoaded = true
        loadModel()
    }

    func onPathChanged() {
        model = nil
        tokenizer = nil
        hasAutoLoaded = false
        loadModel()
    }

    private func loadModel() {
        let path = modelURL?.path ?? modelPath
        guard !path.isEmpty else { return }
        isRunning = true
        statusText = "Loading model..."

        let url = modelURL
        Thread.detachNewThread { [weak self] in
            let dir: URL
            if let url {
                dir = url
                _ = url.startAccessingSecurityScopedResource()
            } else {
                dir = URL(fileURLWithPath: path)
            }
            do {
                let tok = try Tokenizer(from: dir.appendingPathComponent("tokenizer.json"))
                DispatchQueue.main.async { [weak self] in self?.statusText = "Loading weights..." }
                let m = try LlamaModel(directory: dir)
                DispatchQueue.main.async { [weak self] in
                    self?.model = m
                    self?.tokenizer = tok
                    self?.statusText = "Model loaded (\(m.config.numHiddenLayers)L, \(m.config.numAttentionHeads)H)"
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
        let fastGemm = useFastGEMM
        let temp = temperature
        isRunning = true
        liveOutput = ""
        statusText = "Preparing..."

        let model = self.model
        let tokenizer = self.tokenizer

        Thread.detachNewThread { [weak self] in
            self?.generateOnThread(text: text, path: path, flash: flash, fastGemm: fastGemm, temp: temp,
                                   modelURL: url,
                                   existingModel: model, existingTokenizer: tokenizer)
        }
    }

    private nonisolated func generateOnThread(text: String, path: String, flash: Bool, fastGemm: Bool, temp: Float,
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
        model.useFastGEMM = fastGemm

        let tokens = tokenizer.chatTokens(for: text)
        var log = "prompt(\(tokens.count)): \(tokens)\n"
        DispatchQueue.main.async { [weak self] in
            self?.statusText = "Generating (\(tokens.count) prompt tokens)..."
            self?.liveOutput = ""
            self?.debugLog = ""
        }

        let start = CFAbsoluteTimeGetCurrent()
        var firstTokenTime: CFAbsoluteTime?
        var count = 0
        var fullOutput = ""

        model.generate(prompt: tokens, maxTokens: 256, temperature: temp) { tokenId, _ in
            let now = CFAbsoluteTimeGetCurrent()
            if firstTokenTime == nil {
                firstTokenTime = now
                let prefillMs = (now - start) * 1000
                log += String(format: "prefill: %.1fms\n", prefillMs)
            }
            if tokenId == tokenizer.eosToken || tokenId == tokenizer.eotId { return false }
            if self._shouldStop { return false }

            let decoded = tokenizer.decode(tokenId)
            fullOutput += decoded
            count += 1

            if count <= 15 {
                log += "  tok \(count-1): id=\(tokenId) \(repr(decoded))\n"
            }

            let snapshot = fullOutput
            let logSnapshot = log
            DispatchQueue.main.async { [weak self] in
                self?.liveOutput = snapshot
                self?.debugLog = logSnapshot
            }
            return true
        }

        let end = CFAbsoluteTimeGetCurrent()
        let elapsed = end - start
        let ttft = firstTokenTime.map { $0 - start } ?? elapsed
        let decodeTime = firstTokenTime.map { end - $0 } ?? elapsed
        let tps = count > 1 ? Double(count - 1) / decodeTime : (count > 0 ? Double(count) / decodeTime : 0)
        let prefillTps = ttft > 0 ? Double(tokens.count) / ttft : 0
        log += String(format: "prefill: %d tok, %.1f tok/s\n", tokens.count, prefillTps)
        log += String(format: "decode: %d tok, %.1f tok/s\n", count, tps)

        let result = GenerationResult(
            prompt: text, output: fullOutput,
            tokenCount: count, elapsed: elapsed,
            ttft: ttft, tokPerSec: tps, prefillTokPerSec: prefillTps,
            usedFlash: flash)

        let finalLog = log
        DispatchQueue.main.async { [weak self] in
            self?.results.insert(result, at: 0)
            self?.debugLog = finalLog
            self?.statusText = String(format: "P:%.0f D:%.1f tok/s TTFT:%.2fs", prefillTps, tps, ttft)
            self?.isRunning = false
        }
    }
}

private func repr(_ s: String) -> String {
    "'\(s.replacingOccurrences(of: "\n", with: "\\n"))'"
}
