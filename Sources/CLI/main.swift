// swift-llm CLI — pure Swift LLM inference on Apple GPU.

import Foundation
import SwiftLLM

func main() {
    guard CommandLine.arguments.count >= 2 else {
        print("Usage: swift-llm-cli <model-directory> [prompt]")
        print("  model-directory: path to HuggingFace model (with config.json, tokenizer.json, *.safetensors)")
        print("  prompt: text prompt (default: interactive mode)")
        return
    }

    let modelDir = URL(fileURLWithPath: CommandLine.arguments[1])
    let prompt = CommandLine.arguments.count >= 3
        ? CommandLine.arguments.dropFirst(2).joined(separator: " ")
        : nil

    print("=== START ==="); fflush(stdout)
    print("Loading tokenizer..."); fflush(stdout)
    let tokenizer: Tokenizer
    do {
        tokenizer = try Tokenizer(from: modelDir.appendingPathComponent("tokenizer.json"))
        print("  Vocab size: \(tokenizer.vocab.count)"); fflush(stdout)
    } catch {
        print("Failed to load tokenizer: \(error)")
        return
    }

    print("Loading model from \(modelDir.path)..."); fflush(stdout)
    let model: LlamaModel
    do {
        model = try LlamaModel(directory: modelDir)
        print("  Layers: \(model.config.numHiddenLayers)")
        print("  Heads: \(model.config.numAttentionHeads) Q, \(model.config.numKeyValueHeads) KV")
        print("  Hidden: \(model.config.hiddenSize), Head dim: \(model.config.headDim)")
    } catch {
        print("Failed to load model: \(error)")
        return
    }

    func run(text: String) {
        print("Encoding..."); fflush(stdout)
        let tokens = tokenizer.chatTokens(for: text)
        print("Prompt: \(tokens.count) tokens"); fflush(stdout)
        print("---")

        let start = CFAbsoluteTimeGetCurrent()
        var firstTokenTime: CFAbsoluteTime?
        var totalTokens = 0

        model.generate(prompt: tokens, maxTokens: 256) { tokenId, _ in
            if firstTokenTime == nil {
                firstTokenTime = CFAbsoluteTimeGetCurrent()
                let ttft = firstTokenTime! - start
                print("[TTFT: \(String(format: "%.2f", ttft))s]", terminator: " ")
            }

            if tokenId == tokenizer.eosToken || tokenId == tokenizer.eotId { return false }

            let text = tokenizer.decode(tokenId)
            print(text, terminator: "")
            fflush(stdout)
            totalTokens += 1
            return true
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let tps = Double(totalTokens) / elapsed
        print("\n---")
        print("\(totalTokens) tokens in \(String(format: "%.2f", elapsed))s (\(String(format: "%.1f", tps)) tok/s)")
    }

    if let prompt {
        run(text: prompt)
    } else {
        // Interactive mode
        print("Enter prompt (Ctrl+D to quit):")
        while let line = readLine() {
            guard !line.isEmpty else { continue }
            run(text: line)
            print("\nEnter prompt (Ctrl+D to quit):")
        }
    }
}

main()
