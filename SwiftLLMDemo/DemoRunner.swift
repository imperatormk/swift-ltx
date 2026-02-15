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

    /// Conversation history for multi-turn chat with KV cache reuse.
    private var chatHistory: [(role: String, content: String)] = []
    /// Exact token IDs already in the KV cache (avoids BPE round-trip mismatch).
    private var cachedTokenIds: [Int] = []

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
        chatHistory = []
        cachedTokenIds = []
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

        // Add user message to chat history
        chatHistory.append((role: "user", content: text))
        let history = chatHistory
        let cached = cachedTokenIds

        let model = self.model
        let tokenizer = self.tokenizer

        Thread.detachNewThread { [weak self] in
            self?.generateOnThread(text: text, path: path, flash: flash, fastGemm: fastGemm, temp: temp,
                                   modelURL: url,
                                   existingModel: model, existingTokenizer: tokenizer,
                                   chatHistory: history, cachedTokenIds: cached)
        }
    }

    func clearChat() {
        chatHistory = []
        cachedTokenIds = []
        model?.resetCache()
        results = []
    }

    private nonisolated func generateOnThread(text: String, path: String, flash: Bool, fastGemm: Bool, temp: Float,
                                               modelURL url: URL?,
                                               existingModel: LlamaModel?,
                                               existingTokenizer: Tokenizer?,
                                               chatHistory: [(role: String, content: String)],
                                               cachedTokenIds: [Int]) {
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

        // Build token sequence: cached prefix + new user message tokens
        let cachedCount = cachedTokenIds.count
        let fullTokens: [Int]
        if cachedCount > 0 {
            // Continuation: append only the new user turn (no re-tokenization of old turns)
            let newPart = tokenizer.continuationTokens(userMessage: text)
            fullTokens = cachedTokenIds + newPart
        } else {
            // First turn: tokenize full conversation from scratch
            fullTokens = tokenizer.chatTokens(for: chatHistory)
        }
        let newTokens = fullTokens.count - cachedCount
        var log = "total(\(fullTokens.count)) new(\(newTokens)) cached(\(cachedCount))\n"
        DispatchQueue.main.async { [weak self] in
            self?.statusText = "Generating (prefill \(newTokens) new tokens)..."
            self?.liveOutput = ""
            self?.debugLog = ""
        }

        let start = CFAbsoluteTimeGetCurrent()
        var firstTokenTime: CFAbsoluteTime?
        var count = 0
        var fullOutput = ""
        var generatedIds: [Int] = []

        model.generate(prompt: fullTokens, maxTokens: 256, temperature: temp, continueFrom: cachedCount) { tokenId, _ in
            let now = CFAbsoluteTimeGetCurrent()
            if firstTokenTime == nil {
                firstTokenTime = now
                let prefillMs = (now - start) * 1000
                log += String(format: "prefill: %.1fms (%d new tok)\n", prefillMs, newTokens)
            }
            if tokenId == tokenizer.eosToken || tokenId == tokenizer.eotId { return false }
            if self._shouldStop { return false }

            let decoded = tokenizer.decode(tokenId)
            fullOutput += decoded
            generatedIds.append(tokenId)
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
        let prefillTps = newTokens > 0 && ttft > 0 ? Double(newTokens) / ttft : 0
        log += String(format: "prefill: %d new tok, %.1f tok/s\n", newTokens, prefillTps)
        log += String(format: "decode: %d tok, %.1f tok/s\n", count, tps)
        log += String(format: "cache: %d / %d tokens\n", model.cachePosition, model.maxSeqLen)

        let result = GenerationResult(
            prompt: text, output: fullOutput,
            tokenCount: count, elapsed: elapsed,
            ttft: ttft, tokPerSec: tps, prefillTokPerSec: prefillTps,
            usedFlash: flash)

        let finalLog = log
        // Build exact token ID sequence that's now in cache
        let newCachedIds = fullTokens + generatedIds
        let newCachedCount = newCachedIds.count
        DispatchQueue.main.async { [weak self] in
            self?.results.insert(result, at: 0)
            self?.debugLog = finalLog
            self?.statusText = String(format: "P:%.0f D:%.1f tok/s cache:%d/%d", prefillTps, tps, newCachedCount, model.maxSeqLen)
            self?.isRunning = false
            // Add assistant response to history and store exact token IDs
            self?.chatHistory.append((role: "assistant", content: fullOutput))
            self?.cachedTokenIds = newCachedIds
        }
    }
}

private func repr(_ s: String) -> String {
    "'\(s.replacingOccurrences(of: "\n", with: "\\n"))'"
}
