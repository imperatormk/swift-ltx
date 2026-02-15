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

    // Llama 3 pre-tokenizer regex (matches the HuggingFace tokenizer.json pattern).
    // Splits text into words/numbers/punctuation chunks before BPE.
    private static let pretokPattern: NSRegularExpression = {
        let pattern = #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
        return try! NSRegularExpression(pattern: pattern)
    }()

    /// Encode text to token IDs using byte-level BPE with pre-tokenization.
    public func encode(_ text: String) -> [Int] {
        // Pre-tokenize: split text into chunks using the Llama 3 regex
        let nsText = text as NSString
        let matches = Tokenizer.pretokPattern.matches(in: text, range: NSRange(location: 0, length: nsText.length))

        var result: [Int] = []
        for match in matches {
            let chunk = nsText.substring(with: match.range)

            // Map each byte through GPT-2 byte→unicode table
            var tokens: [String] = chunk.utf8.map { byte in
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

            result += tokens.compactMap { vocab[$0] }
        }
        return result
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
        return chatTokens(for: [("user", text)])
    }

    /// Multi-turn chat tokenization. Each message is (role, content).
    /// Returns the full token sequence including system prompt.
    public func chatTokens(for messages: [(role: String, content: String)]) -> [Int] {
        if startHeaderId >= 0 {
            let nl = encode("\n\n")
            let system = encode("system")
            let systemMsg = encode("Cutting Knowledge Date: December 2023\nToday Date: 14 Feb 2026\n\nYou are a helpful assistant.")

            var result = [bosToken, startHeaderId]
            result += system
            result += [endHeaderId]
            result += nl
            result += systemMsg

            for msg in messages {
                result += [eotId, startHeaderId]
                result += encode(msg.role)
                result += [endHeaderId]
                result += nl
                result += encode(msg.content)
            }

            // End with assistant header for generation
            result += [eotId, startHeaderId]
            result += encode("assistant")
            result += [endHeaderId]
            result += nl
            return result
        } else {
            var result = [bosToken]
            for msg in messages {
                result += encode(msg.content)
            }
            return result
        }
    }

    /// Tokenize only the continuation after previous cached tokens.
    /// Produces: [eot] [header:user] content [eot] [header:assistant] \n\n
    public func continuationTokens(userMessage: String) -> [Int] {
        if startHeaderId >= 0 {
            let nl = encode("\n\n")
            var result = [eotId, startHeaderId]
            result += encode("user")
            result += [endHeaderId]
            result += nl
            result += encode(userMessage)
            result += [eotId, startHeaderId]
            result += encode("assistant")
            result += [endHeaderId]
            result += nl
            return result
        } else {
            return encode(userMessage)
        }
    }

    /// Decode a single token.
    public func decode(_ id: Int) -> String {
        guard let tokenStr = reverseVocab[id] else { return "" }
        let bytes: [UInt8] = tokenStr.compactMap { Tokenizer.unicodeToByte[$0] }
        return String(bytes: bytes, encoding: .utf8) ?? String(bytes.map { Character(UnicodeScalar($0)) })
    }
}
