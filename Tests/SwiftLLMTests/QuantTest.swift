// Test quantized ops against MLX reference values.

import XCTest
import Foundation
import Metal
@testable import SwiftLLM

final class QuantTest: XCTestCase {

    let modelDir = URL(fileURLWithPath: "/Users/zimski/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/7f0dc925e0d0afb0322d96f9255cfddf2ba5636e")

    func testEmbeddingQ4() throws {
        let config = try ModelConfig(from: modelDir.appendingPathComponent("config.json"))
        let files = try loadSafeTensors(from: modelDir)
        let weights = try ModelWeights(config: config, files: files)

        // MLX reference: embed[128000][:8]
        let ref: [Float] = [1.526e-05, 1.526e-05, 1.526e-05, 1.526e-05, 1.505e-02, 1.526e-05, 1.526e-05, 1.526e-05]

        let emb: Tensor
        switch weights.embedTokens {
        case .quantized(let w, let s, let b, let K, let gs):
            emb = embeddingQ4(weight: w, scales: s, biases: b, tokenId: 128000, K: K, groupSize: gs)
        case .float(let table):
            emb = embedding(table: table, tokenId: 128000, dim: config.hiddenSize)
        }

        let ptr = emb.buffer.contents().assumingMemoryBound(to: Float.self)
        print("Our embed[128000][:8]:")
        for i in 0..<8 { print(String(format: "  [%d] = %.8f (ref: %.8f)", i, ptr[i], ref[i])) }

        for i in 0..<8 {
            XCTAssertEqual(ptr[i], ref[i], accuracy: 1e-4, "embed mismatch at \(i)")
        }
    }

    func testMatmulQ4() throws {
        let config = try ModelConfig(from: modelDir.appendingPathComponent("config.json"))
        let files = try loadSafeTensors(from: modelDir)
        let weights = try ModelWeights(config: config, files: files)

        // Get embedding for BOS
        let emb: Tensor
        switch weights.embedTokens {
        case .quantized(let w, let s, let b, let K, let gs):
            emb = embeddingQ4(weight: w, scales: s, biases: b, tokenId: 128000, K: K, groupSize: gs)
        case .float(let table):
            emb = embedding(table: table, tokenId: 128000, dim: config.hiddenSize)
        }

        // Run q_proj
        let qOut = linear(emb, weights.layers[0].qProj, M: 1, K: config.hiddenSize, N: config.numAttentionHeads * config.headDim)

        let ptr = qOut.buffer.contents().assumingMemoryBound(to: Float.self)
        // MLX reference: q_proj(embed[128000])[0] = -0.01146
        print("q_proj output[0] = \(ptr[0]) (ref: -0.01146)")
        XCTAssertEqual(ptr[0], -0.01146, accuracy: 0.002, "q_proj[0] mismatch")
    }

    func testRMSNorm() throws {
        let config = try ModelConfig(from: modelDir.appendingPathComponent("config.json"))
        let files = try loadSafeTensors(from: modelDir)
        let weights = try ModelWeights(config: config, files: files)

        // Embed BOS
        let emb: Tensor
        switch weights.embedTokens {
        case .quantized(let w, let s, let b, let K, let gs):
            emb = embeddingQ4(weight: w, scales: s, biases: b, tokenId: 128000, K: K, groupSize: gs)
        case .float(let table):
            emb = embedding(table: table, tokenId: 128000, dim: config.hiddenSize)
        }

        let normed = rmsNorm(emb, weight: weights.layers[0].inputNormWeight, eps: config.rmsNormEps, dim: config.hiddenSize)

        let ptr = normed.buffer.contents().assumingMemoryBound(to: Float.self)
        print("rmsNorm[:8] = \((0..<8).map { String(format: "%.6f", ptr[$0]) }.joined(separator: ", "))")

        for i in 0..<8 {
            XCTAssertFalse(ptr[i].isNaN, "NaN at \(i)")
            XCTAssertFalse(ptr[i].isInfinite, "Inf at \(i)")
        }
    }

    func testLayer0Output() throws {
        let config = try ModelConfig(from: modelDir.appendingPathComponent("config.json"))
        let files = try loadSafeTensors(from: modelDir)
        let weights = try ModelWeights(config: config, files: files)

        // Embed BOS
        let emb: Tensor
        switch weights.embedTokens {
        case .quantized(let w, let s, let b, let K, let gs):
            emb = embeddingQ4(weight: w, scales: s, biases: b, tokenId: 128000, K: K, groupSize: gs)
        case .float(let table):
            emb = embedding(table: table, tokenId: 128000, dim: config.hiddenSize)
        }

        // Manually run layer 0: input_norm → QKV → attn → residual → post_norm → FFN → residual
        let w0 = weights.layers[0]
        let normed = rmsNorm(emb, weight: w0.inputNormWeight, eps: config.rmsNormEps, dim: config.hiddenSize)

        var q = linear(normed, w0.qProj, M: 1, K: config.hiddenSize, N: config.numAttentionHeads * config.headDim)
        var k = linear(normed, w0.kProj, M: 1, K: config.hiddenSize, N: config.numKeyValueHeads * config.headDim)
        let v = linear(normed, w0.vProj, M: 1, K: config.hiddenSize, N: config.numKeyValueHeads * config.headDim)

        // Print pre-RoPE Q
        let qPtr = q.buffer.contents().assumingMemoryBound(to: Float.self)
        print("pre-rope Q[:8] = \((0..<8).map { String(format: "%.6f", qPtr[$0]) }.joined(separator: ", "))")

        q = Tensor(buffer: q.buffer, shape: [config.numAttentionHeads, 1, config.headDim])
        k = Tensor(buffer: k.buffer, shape: [config.numKeyValueHeads, 1, config.headDim])

        q = rope(q, headDim: config.headDim, seqLen: 1, startPos: 0, theta: config.ropeTheta)
        k = rope(k, headDim: config.headDim, seqLen: 1, startPos: 0, theta: config.ropeTheta)

        // Print post-RoPE Q
        let qPtr2 = q.buffer.contents().assumingMemoryBound(to: Float.self)
        print("post-rope Q[:8] = \((0..<8).map { String(format: "%.6f", qPtr2[$0]) }.joined(separator: ", "))")

        // For pos=0, Q@K^T is just the dot product for each head (seq=1), softmax is trivially 1, output=V
        // So attn output = V reshaped, then O projection
        let attnFlat = Tensor(buffer: v.buffer, shape: [1, config.numAttentionHeads * config.headDim])
        // Wait — GQA! v is [numKVHeads, 1, headDim], need to repeat for numAttentionHeads
        // Actually for single token, attn(Q,K,V) where K,V have seq=1: softmax([q@k^T/sqrt(d)]) = [1], so output = V
        // But Q has 24 heads, K/V have 8 heads. Each group of 3 Q heads shares 1 KV head.
        // Output per Q head = V of corresponding KV head.
        let vPtr = v.buffer.contents().assumingMemoryBound(to: Float.self)
        let kvRepeat = config.numAttentionHeads / config.numKeyValueHeads
        var attnOut = [Float](repeating: 0, count: config.numAttentionHeads * config.headDim)
        for h in 0..<config.numAttentionHeads {
            let kvh = h / kvRepeat
            for d in 0..<config.headDim {
                attnOut[h * config.headDim + d] = vPtr[kvh * config.headDim + d]
            }
        }
        let attnTensor = Tensor(attnOut, shape: [1, config.numAttentionHeads * config.headDim])
        let oOut = linear(attnTensor, w0.oProj, M: 1, K: config.numAttentionHeads * config.headDim, N: config.hiddenSize)

        // Residual
        let afterAttn = elemAdd(emb, oOut)

        // FFN
        let normed2 = rmsNorm(afterAttn, weight: w0.postNormWeight, eps: config.rmsNormEps, dim: config.hiddenSize)
        let gate = linear(normed2, w0.gateProj, M: 1, K: config.hiddenSize, N: config.intermediateSize)
        let up = linear(normed2, w0.upProj, M: 1, K: config.hiddenSize, N: config.intermediateSize)
        let activated = elemMul(silu(gate), up)
        let down = linear(activated, w0.downProj, M: 1, K: config.intermediateSize, N: config.hiddenSize)
        let layer0Out = elemAdd(afterAttn, down)

        let lPtr = layer0Out.buffer.contents().assumingMemoryBound(to: Float.self)
        print("layer0[:8] = \((0..<8).map { String(format: "%.6f", lPtr[$0]) }.joined(separator: ", "))")

        // MLX reference
        let ref: [Float] = [0.01009, 0.01101, 0.0921, -0.02014, -0.0094, 0.001762, -0.005432, 0.004852]
        for i in 0..<8 {
            XCTAssertEqual(lPtr[i], ref[i], accuracy: 0.01, "layer0 mismatch at \(i)")
        }
    }

    func testFullForwardBOS() throws {
        let model = try LlamaModel(directory: modelDir)
        let tokenizer = try Tokenizer(from: modelDir.appendingPathComponent("tokenizer.json"))

        // MLX reference top tokens after BOS: 2(#), 14924(Question), 791(The)
        model.generate(prompt: [128000], maxTokens: 1, temperature: 0) { tokenId, _ in
            print("First token: \(tokenId) = '\(tokenizer.decode(tokenId))'")
            let validTopTokens = [2, 14924, 791, 16309, 3936]
            XCTAssert(validTopTokens.contains(tokenId),
                     "Expected one of \(validTopTokens), got \(tokenId) = '\(tokenizer.decode(tokenId))'")
            return false
        }
    }
}
