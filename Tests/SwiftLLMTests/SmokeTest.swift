// Smoke test: build a tiny random-weight Llama and run a forward pass.
// No real model needed — just verifies the full pipeline runs end-to-end.

import XCTest
import Foundation
@testable import SwiftLLM

final class SmokeTest: XCTestCase {

    /// Create a fake model directory with random safetensors + config + tokenizer.
    func testRandomModelForwardPass() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("swift-llm-smoke-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        // Tiny config
        let numLayers = 2
        let hidden = 64
        let intermediate = 128
        let numHeads = 4
        let numKVHeads = 2
        let headDim = hidden / numHeads  // 16
        let vocabSize = 256

        let config: [String: Any] = [
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": numLayers,
            "num_attention_heads": numHeads,
            "num_key_value_heads": numKVHeads,
            "vocab_size": vocabSize,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "head_dim": headDim,
            "tie_word_embeddings": true,
        ]
        let configData = try JSONSerialization.data(withJSONObject: config)
        try configData.write(to: tmpDir.appendingPathComponent("config.json"))

        // Minimal tokenizer.json
        // Build a tiny vocab: single-byte tokens for ASCII 0-255
        var vocab: [String: Int] = [:]
        for i in 0..<vocabSize {
            let ch = String(UnicodeScalar(i == 0 ? 32 : i)!)  // avoid null
            vocab[ch] = i
        }
        let tokenizerJSON: [String: Any] = [
            "model": [
                "type": "BPE",
                "vocab": vocab,
                "merges": [] as [String],
            ] as [String: Any],
            "added_tokens": [
                ["content": "<s>", "id": 1],
                ["content": "</s>", "id": 2],
            ],
        ]
        let tokData = try JSONSerialization.data(withJSONObject: tokenizerJSON)
        try tokData.write(to: tmpDir.appendingPathComponent("tokenizer.json"))

        // Build safetensors with random float32 weights
        let tensors = buildRandomWeights(
            numLayers: numLayers, hidden: hidden, intermediate: intermediate,
            numHeads: numHeads, numKVHeads: numKVHeads, headDim: headDim,
            vocabSize: vocabSize)

        let safetensorsURL = tmpDir.appendingPathComponent("model.safetensors")
        try writeSafeTensors(tensors: tensors, to: safetensorsURL)

        // Load and run
        let model = try LlamaModel(directory: tmpDir)
        XCTAssertEqual(model.config.numHiddenLayers, numLayers)
        XCTAssertEqual(model.config.hiddenSize, hidden)

        // Run generate with a short prompt
        let tokenizer = try Tokenizer(from: tmpDir.appendingPathComponent("tokenizer.json"))
        let prompt = [tokenizer.bosToken, 72, 101, 108, 108, 111]  // <s> H e l l o

        var generated: [Int] = []
        model.generate(prompt: prompt, maxTokens: 5) { tokenId, _ in
            generated.append(tokenId)
            return true
        }

        print("Generated \(generated.count) tokens: \(generated)")
        XCTAssertEqual(generated.count, 5, "Should generate exactly 5 tokens")

        // Tokens should be valid vocab IDs
        for tok in generated {
            XCTAssert(tok >= 0 && tok < vocabSize, "Token \(tok) out of range")
        }
    }

    /// Flash attention and naive attention should produce identical tokens for longer prompts.
    func testFlashVsNaiveLongPrompt() throws {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("swift-llm-longprompt-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmpDir) }

        let numLayers = 2
        let hidden = 64
        let intermediate = 128
        let numHeads = 4
        let numKVHeads = 2
        let headDim = hidden / numHeads
        let vocabSize = 256

        let config: [String: Any] = [
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "num_hidden_layers": numLayers,
            "num_attention_heads": numHeads,
            "num_key_value_heads": numKVHeads,
            "vocab_size": vocabSize,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "head_dim": headDim,
            "tie_word_embeddings": true,
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: tmpDir.appendingPathComponent("config.json"))

        var vocab: [String: Int] = [:]
        for i in 0..<vocabSize {
            vocab[String(UnicodeScalar(i == 0 ? 32 : i)!)] = i
        }
        let tokenizerJSON: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocab, "merges": [] as [String]] as [String: Any],
            "added_tokens": [["content": "<s>", "id": 1], ["content": "</s>", "id": 2]],
        ]
        try JSONSerialization.data(withJSONObject: tokenizerJSON)
            .write(to: tmpDir.appendingPathComponent("tokenizer.json"))

        let tensors = buildRandomWeights(
            numLayers: numLayers, hidden: hidden, intermediate: intermediate,
            numHeads: numHeads, numKVHeads: numKVHeads, headDim: headDim,
            vocabSize: vocabSize)
        try writeSafeTensors(tensors: tensors, to: tmpDir.appendingPathComponent("model.safetensors"))

        // 160 token prompt
        let prompt = [1] + (0..<159).map { _ in Int.random(in: 2..<vocabSize) }

        // Run with naive attention
        let modelNaive = try LlamaModel(directory: tmpDir)
        modelNaive.useFlashAttention = false
        var naiveTokens: [Int] = []
        modelNaive.generate(prompt: prompt, maxTokens: 5, temperature: 0) { tok, _ in
            naiveTokens.append(tok)
            return true
        }

        // Run with flash attention
        let modelFlash = try LlamaModel(directory: tmpDir)
        modelFlash.useFlashAttention = true
        var flashTokens: [Int] = []
        modelFlash.generate(prompt: prompt, maxTokens: 5, temperature: 0) { tok, _ in
            flashTokens.append(tok)
            return true
        }

        print("Naive tokens (160 prompt): \(naiveTokens)")
        print("Flash tokens (160 prompt): \(flashTokens)")
        XCTAssertEqual(naiveTokens, flashTokens, "Flash and naive attention should produce identical tokens for 160-token prompt")
    }
}

// MARK: - Random weight generation + safetensors writer

private func randomFloats(_ count: Int, scale: Float = 0.02) -> [Float] {
    (0..<count).map { _ in Float.random(in: -scale...scale) }
}

private func buildRandomWeights(
    numLayers: Int, hidden: Int, intermediate: Int,
    numHeads: Int, numKVHeads: Int, headDim: Int, vocabSize: Int
) -> [(String, [Int], [Float])] {
    var tensors: [(String, [Int], [Float])] = []

    // Embeddings
    tensors.append(("model.embed_tokens.weight", [vocabSize, hidden],
                     randomFloats(vocabSize * hidden)))

    // Final norm
    tensors.append(("model.norm.weight", [hidden],
                     [Float](repeating: 1.0, count: hidden)))

    for i in 0..<numLayers {
        let p = "model.layers.\(i)"
        let qSize = numHeads * headDim
        let kvSize = numKVHeads * headDim

        // Attention projections
        tensors.append(("\(p).self_attn.q_proj.weight", [qSize, hidden],
                         randomFloats(qSize * hidden)))
        tensors.append(("\(p).self_attn.k_proj.weight", [kvSize, hidden],
                         randomFloats(kvSize * hidden)))
        tensors.append(("\(p).self_attn.v_proj.weight", [kvSize, hidden],
                         randomFloats(kvSize * hidden)))
        tensors.append(("\(p).self_attn.o_proj.weight", [hidden, qSize],
                         randomFloats(hidden * qSize)))

        // FFN
        tensors.append(("\(p).mlp.gate_proj.weight", [intermediate, hidden],
                         randomFloats(intermediate * hidden)))
        tensors.append(("\(p).mlp.up_proj.weight", [intermediate, hidden],
                         randomFloats(intermediate * hidden)))
        tensors.append(("\(p).mlp.down_proj.weight", [hidden, intermediate],
                         randomFloats(hidden * intermediate)))

        // Norms (initialize to 1.0 for stability)
        tensors.append(("\(p).input_layernorm.weight", [hidden],
                         [Float](repeating: 1.0, count: hidden)))
        tensors.append(("\(p).post_attention_layernorm.weight", [hidden],
                         [Float](repeating: 1.0, count: hidden)))
    }

    return tensors
}

/// Write tensors in safetensors format.
private func writeSafeTensors(tensors: [(String, [Int], [Float])], to url: URL) throws {
    // Build header JSON
    var header: [String: Any] = [:]
    var dataOffset = 0
    var allData = Data()

    for (name, shape, values) in tensors {
        let byteCount = values.count * 4
        header[name] = [
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [dataOffset, dataOffset + byteCount],
        ] as [String: Any]
        values.withUnsafeBytes { allData.append(contentsOf: $0) }
        dataOffset += byteCount
    }

    let headerData = try JSONSerialization.data(withJSONObject: header)

    // Pad header to 8-byte alignment
    let headerLen = headerData.count
    let paddedLen = (headerLen + 7) & ~7
    var paddedHeader = headerData
    if paddedLen > headerLen {
        paddedHeader.append(Data(repeating: 0x20, count: paddedLen - headerLen))  // space padding
    }

    // Write: 8-byte LE header length + header JSON + tensor data
    var file = Data()
    var len = UInt64(paddedLen)
    withUnsafeBytes(of: &len) { file.append(contentsOf: $0) }
    file.append(paddedHeader)
    file.append(allData)
    try file.write(to: url)
}
