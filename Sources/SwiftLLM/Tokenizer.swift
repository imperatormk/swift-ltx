// BPE Tokenizer for Llama-family models.
// Reads tokenizer.json (HuggingFace format) — vocab + merges.
// Implements GPT-2 style byte-level BPE (ByteLevel pre-tokenizer).

import Foundation

public struct Tokenizer: Sendable {
    public let vocab: [String: Int]       // token string → id
    public let reverseVocab: [Int: String] // id → token string
    public let mergeRanks: [String: Int]    // "left right" → priority rank
    public let bosToken: Int
    public let eosToken: Int
    public let startHeaderId: Int
    public let endHeaderId: Int
    public let eotId: Int

    // GPT-2 byte ↔ unicode mapping tables
    private static let byteToUnicode: [UInt8: Character] = {
        // Printable ASCII ranges that map to themselves
        var bs: [Int] = []
        bs += Array(0x21...0x7E)  // ! through ~
        bs += Array(0xA1...0xAC)  // ¡ through ¬
        bs += Array(0xAE...0xFF)  // ® through ÿ
        var cs = bs.map { $0 }

        // Everything else maps to 256+n
        var n = 0
        for b in 0..<256 {
            if !bs.contains(b) {
                bs.append(b)
                cs.append(256 + n)
                n += 1
            }
        }

        var table: [UInt8: Character] = [:]
        for (b, c) in zip(bs, cs) {
            table[UInt8(b)] = Character(UnicodeScalar(c)!)
        }
        return table
    }()

    private static let unicodeToByte: [Character: UInt8] = {
        var table: [Character: UInt8] = [:]
        for (b, c) in byteToUnicode {
            table[c] = b
        }
        return table
    }()

    public init(from url: URL) throws {
        let data = try Data(contentsOf: url)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        // Parse vocab from "model" → "vocab"
        let model = json["model"] as! [String: Any]
        let rawVocab = model["vocab"] as! [String: Int]
        self.vocab = rawVocab
        self.reverseVocab = Dictionary(uniqueKeysWithValues: rawVocab.map { ($0.value, $0.key) })

        // Parse merges into rank lookup
        let rawMergesAny = model["merges"] as! [Any]
        var ranks: [String: Int] = [:]
        if let _ = rawMergesAny.first as? String {
            for (i, m) in (rawMergesAny as! [String]).enumerated() {
                ranks[m] = i
            }
        } else {
            for (i, m) in (rawMergesAny as! [[String]]).enumerated() {
                ranks["\(m[0]) \(m[1])"] = i
            }
        }
        self.mergeRanks = ranks

        // Special tokens
        let addedTokens = json["added_tokens"] as? [[String: Any]] ?? []
        var bos = 1, eos = 2, startH = -1, endH = -1, eot = -1
        for tok in addedTokens {
            let content = tok["content"] as? String ?? ""
            let id = tok["id"] as? Int ?? 0
            if content == "<|begin_of_text|>" || content == "<s>" { bos = id }
            if content == "<|end_of_text|>" || content == "</s>" { eos = id }
            if content == "<|start_header_id|>" { startH = id }
            if content == "<|end_header_id|>" { endH = id }
            if content == "<|eot_id|>" { eot = id }
        }
        self.bosToken = bos
        self.eosToken = eos
        self.startHeaderId = startH
        self.endHeaderId = endH
        self.eotId = eot
    }

    /// Encode text to token IDs using byte-level BPE.
    public func encode(_ text: String) -> [Int] {
        // Map each byte through GPT-2 byte→unicode table
        var tokens: [String] = text.utf8.map { byte in
            String(Tokenizer.byteToUnicode[byte]!)
        }

        // BPE: repeatedly find and apply the highest-priority (lowest rank) merge
        while tokens.count >= 2 {
            var bestRank = Int.max
            var bestIdx = -1
            for i in 0..<(tokens.count - 1) {
                let key = "\(tokens[i]) \(tokens[i+1])"
                if let rank = mergeRanks[key], rank < bestRank {
                    bestRank = rank
                    bestIdx = i
                }
            }
            if bestIdx < 0 { break }
            tokens[bestIdx] = tokens[bestIdx] + tokens[bestIdx + 1]
            tokens.remove(at: bestIdx + 1)
        }

        return tokens.compactMap { vocab[$0] }
    }

    /// Decode token IDs back to text.
    public func decode(_ ids: [Int]) -> String {
        let tokenStr = ids.compactMap { reverseVocab[$0] }.joined()
        // Reverse GPT-2 unicode→byte mapping
        let bytes: [UInt8] = tokenStr.compactMap { Tokenizer.unicodeToByte[$0] }
        return String(bytes: bytes, encoding: .utf8) ?? String(bytes.map { Character(UnicodeScalar($0)) })
    }

    /// Wrap user text in Llama 3 chat template tokens.
    public func chatTokens(for text: String) -> [Int] {
        if startHeaderId >= 0 {
            // Llama 3 format
            let userText = encode(text)
            let nl = encode("\n\n")
            let user = encode("user")
            let assistant = encode("assistant")
            var result = [bosToken, startHeaderId]
            result += user
            result += [endHeaderId]
            result += nl
            result += userText
            result += [eotId, startHeaderId]
            result += assistant
            result += [endHeaderId]
            result += nl
            return result
        } else {
            // Fallback: just BOS + text
            return [bosToken] + encode(text)
        }
    }

    /// Decode a single token.
    public func decode(_ id: Int) -> String {
        guard let tokenStr = reverseVocab[id] else { return "" }
        let bytes: [UInt8] = tokenStr.compactMap { Tokenizer.unicodeToByte[$0] }
        return String(bytes: bytes, encoding: .utf8) ?? String(bytes.map { Character(UnicodeScalar($0)) })
    }
}
