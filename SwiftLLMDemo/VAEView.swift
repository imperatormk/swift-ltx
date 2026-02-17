import SwiftUI
import SwiftLLM

struct VAEView: View {
    @StateObject private var runner = VAERunner()
    @State private var showModelPicker = false
    @State private var showLatentPicker = false
    @State private var showDebug = true

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

                        // Model file
                        #if os(macOS)
                        HStack(spacing: 8) {
                            TextField("VAE safetensors path", text: $runner.modelPath)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12, design: .monospaced))
                                .disabled(runner.isRunning)
                            Button("Load") { runner.loadModel() }
                                .buttonStyle(.borderedProminent)
                                .tint(.cyan)
                                .disabled(runner.isRunning || runner.modelPath.isEmpty)
                        }
                        #else
                        Button(action: { showModelPicker = true }) {
                            HStack {
                                Image(systemName: "doc")
                                Text(runner.modelURL?.lastPathComponent ?? "Select VAE Weights (.safetensors)")
                                    .font(.system(size: 13, weight: .medium, design: .monospaced))
                                Spacer()
                            }
                            .foregroundStyle(runner.modelURL != nil ? .white : .gray)
                            .padding(12)
                            .background(Color.white.opacity(0.06))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        .disabled(runner.isRunning)
                        .fileImporter(isPresented: $showModelPicker, allowedContentTypes: [.data]) { result in
                            if case .success(let url) = result {
                                runner.modelURL = url
                                runner.loadModel()
                            }
                        }
                        #endif

                        // Latent file
                        #if os(macOS)
                        HStack(spacing: 8) {
                            TextField("Latent safetensors path", text: $runner.latentPath)
                                .textFieldStyle(.roundedBorder)
                                .font(.system(size: 12, design: .monospaced))
                                .disabled(runner.isRunning)
                        }
                        #else
                        Button(action: { showLatentPicker = true }) {
                            HStack {
                                Image(systemName: "photo")
                                Text(runner.latentURL?.lastPathComponent ?? "Select Latent (.safetensors)")
                                    .font(.system(size: 13, weight: .medium, design: .monospaced))
                                Spacer()
                            }
                            .foregroundStyle(runner.latentURL != nil ? .white : .gray)
                            .padding(12)
                            .background(Color.white.opacity(0.06))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                        .disabled(runner.isRunning)
                        .fileImporter(isPresented: $showLatentPicker, allowedContentTypes: [.data]) { result in
                            if case .success(let url) = result {
                                runner.latentURL = url
                            }
                        }
                        #endif

                        // Debug toggle
                        Toggle(isOn: $showDebug) {
                            HStack(spacing: 6) {
                                Image(systemName: "ladybug")
                                    .foregroundStyle(.yellow)
                                Text("Debug Log")
                                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                                    .foregroundStyle(.white)
                            }
                        }
                        .tint(.yellow)

                        // Status
                        Text(runner.statusText)
                            .font(.system(size: 12, weight: .medium, design: .monospaced))
                            .foregroundStyle(.gray)

                        // Debug log
                        if showDebug, !runner.debugLog.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text("DEBUG")
                                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                                        .foregroundStyle(.yellow)
                                    Spacer()
                                    Button(action: {
                                        #if os(macOS)
                                        NSPasteboard.general.clearContents()
                                        NSPasteboard.general.setString(runner.debugLog, forType: .string)
                                        #else
                                        UIPasteboard.general.string = runner.debugLog
                                        #endif
                                    }) {
                                        Image(systemName: "doc.on.doc")
                                            .font(.system(size: 12))
                                            .foregroundStyle(.yellow)
                                    }
                                    .buttonStyle(.plain)
                                }
                                Text(runner.debugLog)
                                    .font(.system(size: 10, weight: .regular, design: .monospaced))
                                    .foregroundStyle(.yellow.opacity(0.8))
                                    .frame(maxWidth: .infinity, alignment: .leading)
                            }
                            .padding(10)
                            .background(Color.yellow.opacity(0.04))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }

                        // Decoded image
                        if let frame = runner.decodedImage {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("OUTPUT — \(frame.width)x\(frame.height)")
                                    .font(.system(size: 11, weight: .bold, design: .monospaced))
                                    .foregroundStyle(.green)

                                if let img = frameToImage(frame) {
                                    #if os(macOS)
                                    Image(nsImage: img)
                                        .resizable()
                                        .aspectRatio(contentMode: .fit)
                                        .clipShape(RoundedRectangle(cornerRadius: 8))
                                    #else
                                    Image(uiImage: img)
                                        .resizable()
                                        .aspectRatio(contentMode: .fit)
                                        .clipShape(RoundedRectangle(cornerRadius: 8))
                                    #endif
                                }
                            }
                            .padding(12)
                            .background(Color.white.opacity(0.04))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }

                        Spacer(minLength: 20)
                    }
                    .padding(20)
                }

                Divider().overlay(Color.white.opacity(0.1))

                // Decode button
                HStack(spacing: 10) {
                    Spacer()
                    if runner.isRunning {
                        ProgressView()
                            .controlSize(.small)
                        Text("Decoding...")
                            .font(.system(size: 14, weight: .medium, design: .monospaced))
                            .foregroundStyle(.gray)
                    } else {
                        Button(action: { runner.decode() }) {
                            HStack {
                                Image(systemName: "play.fill")
                                Text("Decode")
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.green)
                        .disabled(runner.latentPath.isEmpty && runner.latentURL == nil)

                    }
                    Spacer()
                }
                .padding(16)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(Color(red: 0.08, green: 0.08, blue: 0.10))
            .navigationTitle("VAE Decoder")
            #if os(iOS)
            .navigationBarTitleDisplayMode(.inline)
            #endif
            .onAppear {
                if runner.statusText == "Ready" {
                    runner.loadModel()
                }
            }
        }
    }
}

// MARK: - Image conversion

#if os(macOS)
import AppKit
private func frameToImage(_ frame: DecodedFrame) -> NSImage? {
    let w = frame.width, h = frame.height
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(
              data: UnsafeMutableRawPointer(mutating: (frame.rgbaData as NSData).bytes),
              width: w, height: h,
              bitsPerComponent: 8, bytesPerRow: w * 4,
              space: colorSpace,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
          ),
          let cgImage = ctx.makeImage()
    else { return nil }
    return NSImage(cgImage: cgImage, size: NSSize(width: w, height: h))
}
#else
import UIKit
private func frameToImage(_ frame: DecodedFrame) -> UIImage? {
    let w = frame.width, h = frame.height
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
          let ctx = CGContext(
              data: UnsafeMutableRawPointer(mutating: (frame.rgbaData as NSData).bytes),
              width: w, height: h,
              bitsPerComponent: 8, bytesPerRow: w * 4,
              space: colorSpace,
              bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
          ),
          let cgImage = ctx.makeImage()
    else { return nil }
    return UIImage(cgImage: cgImage)
}
#endif

#Preview {
    VAEView()
        .preferredColorScheme(.dark)
}
