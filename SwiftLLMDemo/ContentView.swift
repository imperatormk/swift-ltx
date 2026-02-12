import SwiftUI
import SwiftLLM

struct ContentView: View {
    @StateObject private var runner = DemoRunner()
    @State private var showFolderPicker = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        // Device info
                        HStack(spacing: 12) {
                            Image(systemName: "cpu")
                                .font(.title2)
                                .foregroundStyle(.cyan)
                            VStack(alignment: .leading, spacing: 2) {
                                Text("GPU")
                                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.gray)
                                Text(runner.deviceName)
                                    .font(.system(size: 15, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(.white)
                            }
                            Spacer()
                        }
                        .padding(14)
                        .background(Color.white.opacity(0.06))
                        .clipShape(RoundedRectangle(cornerRadius: 10))

                        // Model path
                        #if os(macOS)
                        HStack(spacing: 8) {
                            TextField("Model path", text: $runner.modelPath)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12, design: .monospaced))
                                .disabled(runner.isRunning)
                                .onSubmit { runner.onPathChanged() }
                        }
                        #else
                        Button(action: { showFolderPicker = true }) {
                            HStack {
                                Image(systemName: "folder")
                                Text(runner.modelURL?.lastPathComponent ?? "Select Model Folder")
                                    .font(.system(size: 13, weight: .medium, design: .monospaced))
                                Spacer()
                            }
                            .foregroundStyle(runner.modelURL != nil ? .white : .gray)
                            .padding(12)
                            .background(Color.white.opacity(0.06))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        .disabled(runner.isRunning)
                        .fileImporter(isPresented: $showFolderPicker, allowedContentTypes: [.folder]) { result in
                            if case .success(let url) = result {
                                runner.modelURL = url
                                runner.modelPath = url.path
                                runner.onPathChanged()
                            }
                        }
                        #endif

                        // Flash attention toggle
                        Toggle(isOn: $runner.useFlashAttention) {
                            HStack(spacing: 6) {
                                Image(systemName: runner.useFlashAttention ? "bolt.fill" : "square.grid.2x2")
                                    .foregroundStyle(runner.useFlashAttention ? .cyan : .orange)
                                Text(runner.useFlashAttention ? "Flash Attention" : "Naive Attention")
                                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(.white)
                            }
                        }
                        .tint(.cyan)
                        .disabled(runner.isRunning)

                        // Status
                        Text(runner.statusText)
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(.gray)

                        // Live output
                        if !runner.liveOutput.isEmpty || runner.isRunning {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("OUTPUT")
                                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                                    .foregroundStyle(.gray)
                                Text(runner.liveOutput.isEmpty ? "..." : runner.liveOutput)
                                    .font(.system(size: 16, weight: .regular, design: .monospaced))
                                    .foregroundStyle(.green)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .padding(12)
                            .background(Color.white.opacity(0.04))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }

                        // History
                        ForEach(runner.results) { result in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(result.prompt)
                                    .font(.system(size: 12, weight: .medium, design: .monospaced))
                                    .foregroundStyle(.cyan)
                                Text(result.output)
                                    .font(.system(size: 14, weight: .regular, design: .monospaced))
                                    .foregroundStyle(.white)
                                HStack {
                                    Text(result.usedFlash ? "FLASH" : "NAIVE")
                                        .foregroundStyle(result.usedFlash ? .cyan : .orange)
                                    Text("\(result.tokenCount) tok")
                                    Text(String(format: "%.1f tok/s", result.tokPerSec))
                                    Text(String(format: "TTFT %.2fs", result.ttft))
                                }
                                .font(.system(size: 11, weight: .medium, design: .monospaced))
                                .foregroundStyle(.gray)
                            }
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.white.opacity(0.04))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }

                        Spacer(minLength: 20)
                    }
                    .padding(20)
                }

                Divider().overlay(Color.white.opacity(0.1))

                // Input
                HStack(spacing: 10) {
                    TextField("Prompt", text: $runner.prompt)
                        .textFieldStyle(.roundedBorder)
                        .font(.system(size: 14, design: .monospaced))
                        .onSubmit { runner.run() }
                        .disabled(runner.isRunning)

                    if runner.isRunning {
                        Button(action: { runner.stop() }) {
                            Image(systemName: "stop.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.red)
                    } else {
                        Button(action: { runner.run() }) {
                            Image(systemName: "play.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.cyan)
                        .disabled(runner.prompt.isEmpty)
                    }
                }
                .padding(16)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(red: 0.08, green: 0.08, blue: 0.10))
            .navigationTitle("SwiftLLM")
            .onAppear { runner.autoLoadIfNeeded() }
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
        }
    }
}

#Preview {
    ContentView()
        .preferredColorScheme(.dark)
}
