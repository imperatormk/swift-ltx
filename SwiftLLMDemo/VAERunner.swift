import Foundation
import Metal
import CoreGraphics
import SwiftLLM

@MainActor
final class VAERunner: ObservableObject {
    @Published var isRunning = false
    @Published var statusText = "Ready"
    @Published var debugLog = ""
    @Published var decodedImage: DecodedFrame?

    #if os(macOS)
    @Published var modelPath = NSString(string: "~/Downloads/ltxv_vae_decoder_f16.safetensors").expandingTildeInPath
    @Published var latentPath = "/tmp/mlx_vae_input_latents.safetensors"
    #else
    @Published var modelPath = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let vaeDir = docs.appendingPathComponent("vae")
        let modelFile = vaeDir.appendingPathComponent("ltxv_vae_decoder_f16.safetensors")
        if FileManager.default.fileExists(atPath: modelFile.path) { return modelFile.path }
        return ""
    }()
    @Published var latentPath = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let vaeDir = docs.appendingPathComponent("vae")
        let latentFile = vaeDir.appendingPathComponent("ltxv_pipeline_latents.safetensors")
        if FileManager.default.fileExists(atPath: latentFile.path) { return latentFile.path }
        return ""
    }()
    #endif
    @Published var modelURL: URL?
    @Published var latentURL: URL?

    private var decoder: VAEDecoder?

    var deviceName: String {
        MetalContext.shared.device.name
    }

    func loadModel() {
        let url = modelURL
        let path: String
        path = url?.path ?? modelPath
        guard !path.isEmpty else {
            statusText = "Select model file first"
            return
        }
        isRunning = true
        statusText = "Loading VAE weights..."

        Thread.detachNewThread { [weak self] in
            do {
                let fileURL: URL
                if let url {
                    fileURL = url
                    _ = url.startAccessingSecurityScopedResource()
                } else {
                    fileURL = URL(fileURLWithPath: path)
                }
                let file = try SafeTensorsFile(url: fileURL)

                // Detect prefix: check for "decoder." or "vae.decoder." keys
                let prefix: String
                if file.tensors.keys.contains(where: { $0.hasPrefix("decoder.") }) {
                    prefix = "decoder."
                } else if file.tensors.keys.contains(where: { $0.hasPrefix("vae.decoder.") }) {
                    prefix = "vae.decoder."
                } else {
                    throw VAEError.noDecoderKeys
                }

                let dec = try VAEDecoder(file: file, prefix: prefix)
                let keyCount = file.tensors.count
                DispatchQueue.main.async { [weak self] in
                    self?.decoder = dec
                    self?.statusText = "VAE loaded (\(keyCount) tensors)"
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

    func decode() {
        guard !isRunning else { return }
        guard let decoder else {
            statusText = "Load model first"
            return
        }

        let url = latentURL
        let path: String
        path = url?.path ?? latentPath
        guard !path.isEmpty else {
            statusText = "Select latent file first"
            return
        }

        isRunning = true
        statusText = "Decoding..."
        debugLog = ""

        Thread.detachNewThread { [weak self] in
            guard let self else { return }
            do {
                let fileURL: URL
                if let url {
                    fileURL = url
                    _ = url.startAccessingSecurityScopedResource()
                } else {
                    fileURL = URL(fileURLWithPath: path)
                }
                let latentFile = try SafeTensorsFile(url: fileURL)

                // Find the latent tensor (usually "latent" or first tensor)
                let latentKey: String
                if latentFile.tensors.keys.contains("latent") {
                    latentKey = "latent"
                } else if let first = latentFile.tensors.keys.first {
                    latentKey = first
                } else {
                    throw VAEError.noLatentTensor
                }

                let info = latentFile.tensors[latentKey]!
                let ptr = latentFile.pointer(for: latentKey)!
                var log = "Latent: \(latentKey) \(info.shape) \(info.dtype)\n"

                // Create Metal buffer from latent data
                let ctx = MetalContext.shared
                let latent: Tensor
                if info.dtype == .float16 {
                    let buf = ctx.device.makeBuffer(bytes: ptr, length: info.byteCount, options: .storageModeShared)!
                    latent = Tensor(buffer: buf, shape: info.shape, dtype: .float16)
                } else if info.dtype == .float32 {
                    // Convert f32 → f16
                    let count = info.byteCount / 4
                    let src = ptr.assumingMemoryBound(to: Float.self)
                    let buf = ctx.device.makeBuffer(length: count * 2, options: .storageModeShared)!
                    let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
                    for i in 0..<count {
                        var f = src[i]
                        var h: UInt16 = 0
                        withUnsafePointer(to: &f) { fp in
                            let bits = fp.withMemoryRebound(to: UInt32.self, capacity: 1) { $0.pointee }
                            let sign = (bits >> 16) & 0x8000
                            let exp = Int((bits >> 23) & 0xFF) - 127 + 15
                            let frac = bits & 0x7FFFFF
                            if exp <= 0 { h = UInt16(sign) }
                            else if exp >= 31 { h = UInt16(sign | 0x7C00) }
                            else { h = UInt16(sign | UInt32(exp) << 10 | (frac >> 13)) }
                        }
                        dst[i] = h
                    }
                    latent = Tensor(buffer: buf, shape: info.shape, dtype: .float16)
                } else {
                    throw VAEError.unsupportedDtype(String(describing: info.dtype))
                }

                log += "Decoding latent \(info.shape)...\n"
                let logSnap1 = log
                DispatchQueue.main.async { [weak self] in self?.debugLog = logSnap1 }

                let start = CFAbsoluteTimeGetCurrent()
                var stepTimes: [(String, Double)] = []
                var lastTime = start
                let output = decoder.decode(latent) { msg in
                    let now = CFAbsoluteTimeGetCurrent()
                    let dt = now - lastTime
                    lastTime = now
                    let entry = String(format: "[%.2fs +%.3fs] %@", now - start, dt, msg)
                    stepTimes.append((entry, dt))
                    log += entry + "\n"
                    let snap = log
                    DispatchQueue.main.async { [weak self] in self?.debugLog = snap }
                }
                let elapsed = CFAbsoluteTimeGetCurrent() - start

                // output shape: [B, C=3, F, H, W] f16
                log += String(format: "Decode time: %.2fs\n", elapsed)
                log += "Output shape: \(output.shape)\n"

                // Extract first frame as image pixels
                let frame = extractFirstFrame(from: output)
                log += "Frame: \(frame.width)x\(frame.height)\n"

                let finalLog = log
                DispatchQueue.main.async { [weak self] in
                    self?.decodedImage = frame
                    self?.debugLog = finalLog
                    self?.statusText = String(format: "Decoded in %.2fs — %dx%d", elapsed, frame.width, frame.height)
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
}

// MARK: - Helpers

struct DecodedFrame {
    let width: Int
    let height: Int
    let rgbaData: Data  // width*height*4 bytes, RGBA8
}

enum VAEError: Error, CustomStringConvertible {
    case noDecoderKeys
    case noLatentTensor
    case unsupportedDtype(String)

    var description: String {
        switch self {
        case .noDecoderKeys: return "No decoder keys found in safetensors"
        case .noLatentTensor: return "No latent tensor found in safetensors"
        case .unsupportedDtype(let d): return "Unsupported dtype: \(d)"
        }
    }
}

/// Extract the first frame (F=0) from a [B, 3, F, H, W] f16 tensor → RGBA8 Data.
func extractFirstFrame(from tensor: Tensor) -> DecodedFrame {
    let shape = tensor.shape
    print("[extractFirstFrame] shape=\(shape) count=\(tensor.count) bufferLen=\(tensor.buffer.length)")
    // [B, C, F, H, W]
    guard shape.count >= 5 else {
        print("[extractFirstFrame] ERROR: expected 5D, got \(shape)")
        return DecodedFrame(width: 1, height: 1, rgbaData: Data([255, 0, 0, 255]))
    }
    let C = shape[1]
    let F = shape[2]
    let H = shape[3]
    let W = shape[4]

    let totalElements = shape.reduce(1, *)
    let bufferElements = tensor.buffer.length / 2  // f16
    guard bufferElements >= totalElements else {
        print("[extractFirstFrame] ERROR: buffer too small: \(bufferElements) < \(totalElements)")
        return DecodedFrame(width: 1, height: 1, rgbaData: Data([255, 0, 0, 255]))
    }

    let ptr = tensor.buffer.contents().assumingMemoryBound(to: UInt16.self)
    // Layout: B * C * F * H * W, take B=0, F=0
    let chanStride = F * H * W

    var rgba = Data(count: H * W * 4)
    rgba.withUnsafeMutableBytes { raw in
        let dst = raw.assumingMemoryBound(to: UInt8.self).baseAddress!
        for y in 0..<H {
            for x in 0..<W {
                let spatialIdx = y * W + x
                for c in 0..<min(C, 3) {
                    let srcIdx = c * chanStride + spatialIdx
                    let f16bits = ptr[srcIdx]
                    let val = f16ToFloat(f16bits)
                    let mapped = val / 2.0 + 0.5
                    let clamped = mapped.isNaN ? Float(0) : min(max(mapped, 0), 1)
                    dst[spatialIdx * 4 + c] = UInt8(clamped * 255)
                }
                dst[spatialIdx * 4 + 3] = 255
            }
        }
    }
    return DecodedFrame(width: W, height: H, rgbaData: rgba)
}

/// Extract ALL frames from a [B, 3, F, H, W] f16 tensor → [DecodedFrame].
func extractAllFrames(from tensor: Tensor, targetWidth: Int? = nil, targetHeight: Int? = nil) -> [DecodedFrame] {
    let shape = tensor.shape
    guard shape.count >= 5 else { return [extractFirstFrame(from: tensor)] }
    let C = shape[1], F = shape[2], H = shape[3], W = shape[4]
    let totalElements = shape.reduce(1, *)
    let bufferElements = tensor.buffer.length / 2
    guard bufferElements >= totalElements else { return [extractFirstFrame(from: tensor)] }

    let ptr = tensor.buffer.contents().assumingMemoryBound(to: UInt16.self)
    let chanStride = F * H * W
    let frameStride = H * W

    let tW = targetWidth ?? W
    let tH = targetHeight ?? H
    let needsResize = (tW != W || tH != H)

    var frames = [DecodedFrame]()
    for f in 0..<F {
        var rgba = Data(count: H * W * 4)
        rgba.withUnsafeMutableBytes { raw in
            let dst = raw.assumingMemoryBound(to: UInt8.self).baseAddress!
            for y in 0..<H {
                for x in 0..<W {
                    let spatialIdx = y * W + x
                    for c in 0..<min(C, 3) {
                        let srcIdx = c * chanStride + f * frameStride + spatialIdx
                        let val = Float(Float16(bitPattern: ptr[srcIdx]))
                        let mapped = val / 2.0 + 0.5
                        let clamped = mapped.isNaN ? Float(0) : min(max(mapped, 0), 1)
                        dst[spatialIdx * 4 + c] = UInt8(clamped * 255)
                    }
                    dst[spatialIdx * 4 + 3] = 255
                }
            }
        }

        if needsResize {
            // Bilinear resize using CoreGraphics (matches Python F.interpolate bilinear)
            let resized = bilinearResize(rgba, srcW: W, srcH: H, dstW: tW, dstH: tH)
            frames.append(DecodedFrame(width: tW, height: tH, rgbaData: resized))
        } else {
            frames.append(DecodedFrame(width: W, height: H, rgbaData: rgba))
        }
    }
    return frames
}

/// Bilinear resize RGBA8 data using CoreGraphics.
private func bilinearResize(_ data: Data, srcW: Int, srcH: Int, dstW: Int, dstH: Int) -> Data {
    guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return data }
    var mutData = data
    guard let srcCtx = mutData.withUnsafeMutableBytes({ raw -> CGContext? in
        CGContext(data: raw.baseAddress, width: srcW, height: srcH,
                  bitsPerComponent: 8, bytesPerRow: srcW * 4,
                  space: colorSpace,
                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
    }), let srcImage = srcCtx.makeImage() else { return data }

    var dstData = Data(count: dstW * dstH * 4)
    dstData.withUnsafeMutableBytes { raw in
        guard let dstCtx = CGContext(data: raw.baseAddress, width: dstW, height: dstH,
                                     bitsPerComponent: 8, bytesPerRow: dstW * 4,
                                     space: colorSpace,
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return }
        dstCtx.interpolationQuality = .high
        dstCtx.draw(srcImage, in: CGRect(x: 0, y: 0, width: dstW, height: dstH))
    }
    return dstData
}

private func f16ToFloat(_ bits: UInt16) -> Float {
    return Float(Float16(bitPattern: bits))
}
