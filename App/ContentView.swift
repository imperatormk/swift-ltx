import SwiftUI
import SwiftLLM

struct ContentView: View {
    @StateObject private var vm = InferenceViewModel()

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Output area
                ScrollViewReader { proxy in
                    ScrollView {
                        VStack(alignment: .leading, spacing: 8) {
                            if vm.status != .idle {
                                Text(vm.statusText)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .padding(.horizontal)
                            }

                            Text(vm.output)
                                .font(.system(.body, design: .monospaced))
                                .padding(.horizontal)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .id("output")

                            if vm.stats.isEmpty == false {
                                Text(vm.stats)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .padding(.horizontal)
                            }
                        }
                        .padding(.vertical)
                    }
                    .onChange(of: vm.output) {
                        withAnimation {
                            proxy.scrollTo("output", anchor: .bottom)
                        }
                    }
                }

                Divider()

                // Input area
                HStack {
                    TextField("Enter prompt...", text: $vm.prompt)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { vm.run() }
                        .disabled(vm.status == .generating)

                    Button(vm.status == .generating ? "..." : "Go") {
                        vm.run()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(vm.prompt.isEmpty || vm.status == .generating)
                }
                .padding()
            }
            .navigationTitle("SwiftLLM")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
        }
    }
}

// MARK: - ViewModel

enum InferenceStatus {
    case idle, loading, generating
}

@MainActor
class InferenceViewModel: ObservableObject {
    @Published var prompt: String = "Hello world"
    @Published var output: String = ""
    @Published var stats: String = ""
    @Published var statusText: String = ""
    @Published var status: InferenceStatus = .idle

    /// We use a tiny random-weight model for demo.
    /// Replace with real model path for actual inference.
    private var model: LlamaModel?
    private var tokenizer: Tokenizer?

    func run() {
        guard status != .generating else { return }
        let text = prompt
        prompt = ""
        output = ""
        stats = ""
        status = .loading
        statusText = "Preparing model..."

        Task.detached { [weak self] in
            await self?.generate(text: text)
        }
    }

    private func generate(text: String) async {
        // Build random model on first run (or reuse)
        if model == nil {
            await MainActor.run { statusText = "Building random model..." }
            do {
                let dir = try Self.createRandomModel()
                let m = try LlamaModel(directory: dir)
                let t = try Tokenizer(from: dir.appendingPathComponent("tokenizer.json"))
                await MainActor.run {
                    self.model = m
                    self.tokenizer = t
                }
            } catch {
                await MainActor.run {
                    self.output = "Error: \(error)"
                    self.status = .idle
                }
                return
            }
        }

        guard let model = await model, let tokenizer = await tokenizer else { return }

        let tokens = [tokenizer.bosToken] + tokenizer.encode(text)
        await MainActor.run {
            statusText = "Generating (\(tokens.count) prompt tokens)..."
            status = .generating
            output = ""
        }

        let start = CFAbsoluteTimeGetCurrent()
        var firstTokenTime: CFAbsoluteTime?
        var count = 0

        model.generate(prompt: tokens, maxTokens: 50) { tokenId, _ in
            if firstTokenTime == nil {
                firstTokenTime = CFAbsoluteTimeGetCurrent()
            }
            if tokenId == tokenizer.eosToken { return false }

            let decoded = tokenizer.decode(tokenId)
            count += 1

            DispatchQueue.main.async { [weak self] in
                self?.output += decoded
            }
            return true
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let ttft = firstTokenTime.map { $0 - start } ?? elapsed
        let tps = count > 0 ? Double(count) / elapsed : 0

        await MainActor.run {
            self.stats = "\(count) tokens · \(String(format: "%.2f", elapsed))s · \(String(format: "%.1f", tps)) tok/s · TTFT \(String(format: "%.2f", ttft))s"
            self.statusText = "Done"
            self.status = .idle
        }
    }

    /// Create a tiny random-weight model in tmp for demo purposes.
    private static func createRandomModel() throws -> URL {
        let tmpDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("swift-llm-demo")

        // Reuse if already created
        if FileManager.default.fileExists(atPath: tmpDir.appendingPathComponent("config.json").path) {
            return tmpDir
        }

        try FileManager.default.createDirectory(at: tmpDir, withIntermediateDirectories: true)

        let numLayers = 2
        let hidden = 64
        let intermediate = 128
        let numHeads = 4
        let numKVHeads = 2
        let headDim = hidden / numHeads
        let vocabSize = 256

        // config.json
        let config: [String: Any] = [
            "hidden_size": hidden, "intermediate_size": intermediate,
            "num_hidden_layers": numLayers, "num_attention_heads": numHeads,
            "num_key_value_heads": numKVHeads, "vocab_size": vocabSize,
            "max_position_embeddings": 512, "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0, "head_dim": headDim,
            "tie_word_embeddings": true,
        ]
        try JSONSerialization.data(withJSONObject: config)
            .write(to: tmpDir.appendingPathComponent("config.json"))

        // tokenizer.json
        var vocab: [String: Int] = [:]
        for i in 0..<vocabSize {
            vocab[String(UnicodeScalar(i == 0 ? 32 : i)!)] = i
        }
        let tok: [String: Any] = [
            "model": ["type": "BPE", "vocab": vocab, "merges": [] as [String]] as [String: Any],
            "added_tokens": [["content": "<s>", "id": 1], ["content": "</s>", "id": 2]],
        ]
        try JSONSerialization.data(withJSONObject: tok)
            .write(to: tmpDir.appendingPathComponent("tokenizer.json"))

        // safetensors
        var tensors: [(String, [Int], [Float])] = []
        func rand(_ n: Int) -> [Float] { (0..<n).map { _ in Float.random(in: -0.02...0.02) } }
        func ones(_ n: Int) -> [Float] { [Float](repeating: 1.0, count: n) }

        tensors.append(("model.embed_tokens.weight", [vocabSize, hidden], rand(vocabSize * hidden)))
        tensors.append(("model.norm.weight", [hidden], ones(hidden)))

        for i in 0..<numLayers {
            let p = "model.layers.\(i)"
            let qSz = numHeads * headDim, kvSz = numKVHeads * headDim
            tensors.append(("\(p).self_attn.q_proj.weight", [qSz, hidden], rand(qSz * hidden)))
            tensors.append(("\(p).self_attn.k_proj.weight", [kvSz, hidden], rand(kvSz * hidden)))
            tensors.append(("\(p).self_attn.v_proj.weight", [kvSz, hidden], rand(kvSz * hidden)))
            tensors.append(("\(p).self_attn.o_proj.weight", [hidden, qSz], rand(hidden * qSz)))
            tensors.append(("\(p).mlp.gate_proj.weight", [intermediate, hidden], rand(intermediate * hidden)))
            tensors.append(("\(p).mlp.up_proj.weight", [intermediate, hidden], rand(intermediate * hidden)))
            tensors.append(("\(p).mlp.down_proj.weight", [hidden, intermediate], rand(hidden * intermediate)))
            tensors.append(("\(p).input_layernorm.weight", [hidden], ones(hidden)))
            tensors.append(("\(p).post_attention_layernorm.weight", [hidden], ones(hidden)))
        }

        // Write safetensors
        var header: [String: Any] = [:]
        var dataOffset = 0
        var allData = Data()
        for (name, shape, values) in tensors {
            let bytes = values.count * 4
            header[name] = ["dtype": "F32", "shape": shape, "data_offsets": [dataOffset, dataOffset + bytes]] as [String: Any]
            values.withUnsafeBytes { allData.append(contentsOf: $0) }
            dataOffset += bytes
        }
        let headerData = try JSONSerialization.data(withJSONObject: header)
        let padLen = (headerData.count + 7) & ~7
        var padded = headerData
        if padLen > headerData.count { padded.append(Data(repeating: 0x20, count: padLen - headerData.count)) }
        var file = Data()
        var len = UInt64(padLen)
        withUnsafeBytes(of: &len) { file.append(contentsOf: $0) }
        file.append(padded)
        file.append(allData)
        try file.write(to: tmpDir.appendingPathComponent("model.safetensors"))

        return tmpDir
    }
}
