// Llama model configuration — parsed from config.json in HuggingFace format.

import Foundation

public struct ModelConfig: Codable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let vocabSize: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let headDim: Int
    public let tieWordEmbeddings: Bool
    public let quantization: QuantizationConfig?

    public struct QuantizationConfig: Codable {
        public let groupSize: Int
        public let bits: Int

        enum CodingKeys: String, CodingKey {
            case groupSize = "group_size"
            case bits
        }
    }

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case vocabSize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
        case quantization
    }

    public init(from url: URL) throws {
        let data = try Data(contentsOf: url)
        self = try JSONDecoder().decode(ModelConfig.self, from: data)
    }

    /// GQA repeat factor: how many Q heads share each KV head.
    public var kvRepeatFactor: Int { numAttentionHeads / numKeyValueHeads }
}
