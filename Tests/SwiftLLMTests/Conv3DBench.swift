// Conv3DBench.swift — Compare im2col conv3d vs implicit GEMM conv3d
// Run: swift test --filter Conv3DBench

import XCTest
import Foundation
import Metal
@testable import SwiftLLM

final class Conv3DBench: XCTestCase {

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Allocate a shared MTLBuffer filled with small f32 values cast to f16 via Float16.
    private func randF16(_ n: Int, seed: UInt64 = 42) -> MTLBuffer {
        let buf = MetalContext.shared.device
            .makeBuffer(length: max(n * 2, 4), options: .storageModeShared)!
        let ptr = buf.contents().assumingMemoryBound(to: Float16.self)
        var rng = seed
        for i in 0..<n {
            rng &+= 0x9e3779b97f4a7c15
            rng = (rng ^ (rng >> 30)) &* 0xbf58476d1ce4e5b9
            rng = (rng ^ (rng >> 27)) &* 0x94d049bb133111eb
            rng ^= rng >> 31
            let f = Float(Int64(bitPattern: rng) % 1000) * 0.0001
            ptr[i] = Float16(f)
        }
        return buf
    }

    /// Read f16 buffer as [Float].
    private func readF16(_ buf: MTLBuffer, count: Int) -> [Float] {
        let ptr = buf.contents().assumingMemoryBound(to: Float16.self)
        return (0..<count).map { Float(ptr[$0]) }
    }

    /// Max absolute difference between two float arrays.
    private func maxAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        zip(a, b).map { abs($0 - $1) }.max() ?? 0
    }

    /// Mean absolute difference.
    private func meanAbsDiff(_ a: [Float], _ b: [Float]) -> Float {
        let s = zip(a, b).map { abs($0 - $1) }.reduce(0, +)
        return s / Float(a.count)
    }

    // ── correctness test ─────────────────────────────────────────────────────

    /// Run both kernels on same inputs, compare outputs.
    private func runCorrectness(
        B: Int, C_in: Int, D: Int, H: Int, W: Int,
        C_out: Int, kD: Int, kH: Int, kW: Int,
        sD: Int = 1, sH: Int = 1, sW: Int = 1,
        pD: Int = 0, pH: Int = 0, pW: Int = 0,
        label: String
    ) {
        let inputN  = B * C_in * D * H * W
        let weightN = C_out * C_in * kD * kH * kW
        let biasN   = C_out
        let D_out = (D + 2*pD - kD) / sD + 1
        let H_out = (H + 2*pH - kH) / sH + 1
        let W_out = (W + 2*pW - kW) / sW + 1
        let outN = B * C_out * D_out * H_out * W_out

        let xBuf  = randF16(inputN,  seed: 1)
        let wBuf  = randF16(weightN, seed: 2)
        let bBuf  = randF16(biasN,   seed: 3)

        let xRef  = Tensor(buffer: xBuf, shape: [B, C_in, D, H, W],             dtype: .float16)
        let wRef  = Tensor(buffer: wBuf, shape: [C_out, C_in, kD, kH, kW],      dtype: .float16)
        let bRef  = Tensor(buffer: bBuf, shape: [C_out],                         dtype: .float16)

        // ── reference: existing im2col path ──────────────────────────────────
        let refOut = conv3d(xRef, weight: wRef, bias: bRef,
                            B: B, C_in: C_in, D: D, H: H, W: W,
                            C_out: C_out, kD: kD, kH: kH, kW: kW,
                            sD: sD, sH: sH, sW: sW,
                            pD: pD, pH: pH, pW: pW)
        let refVals = readF16(refOut.buffer, count: outN)

        // ── candidate: implicit GEMM ──────────────────────────────────────────
        let candOut = conv3d_implicit(xRef, weight: wRef, bias: bRef,
                                      B: B, C_in: C_in, D: D, H: H, W: W,
                                      C_out: C_out, kD: kD, kH: kH, kW: kW,
                                      sD: sD, sH: sH, sW: sW,
                                      pD: pD, pH: pH, pW: pW)
        let candVals = readF16(candOut.buffer, count: outN)

        let maxD = maxAbsDiff(refVals, candVals)
        let meanD = meanAbsDiff(refVals, candVals)
        print(String(format: "  %-40@  maxΔ=%.5f  meanΔ=%.6f  %@",
                     label, maxD, meanD, maxD < 0.01 ? "✓ PASS" : "✗ FAIL"))
        XCTAssertLessThan(maxD, 0.01, "\(label): max abs diff \(maxD) too large")
    }

    func testCorrectness() {
        print("\n── Conv3D correctness: im2col vs implicit GEMM ──────────────────────")

        // 1x1x1 pointwise (shared path, should be identical)
        runCorrectness(B:1, C_in:128, D:8,  H:32, W:32,  C_out:128,  kD:1, kH:1, kW:1, label:"1x1x1 B1 C128 8x32x32")

        // 3x3x3 small
        runCorrectness(B:1, C_in:32,  D:4,  H:8,  W:8,   C_out:64,   kD:3, kH:3, kW:3, pD:1, pH:1, pW:1, label:"3x3x3 B1 C32->64 4x8x8")

        // 3x3x3 LTX conv_in size: [1, 128, 2, 48, 48]
        runCorrectness(B:1, C_in:128, D:2,  H:48, W:48,  C_out:1024, kD:3, kH:3, kW:3, pD:1, pH:1, pW:1, label:"3x3x3 B1 C128->1024 2x48x48")

        // 3x3x3 mid-block: [1, 1024, 2, 48, 48]
        runCorrectness(B:1, C_in:1024,D:2,  H:48, W:48,  C_out:1024, kD:3, kH:3, kW:3, pD:1, pH:1, pW:1, label:"3x3x3 B1 C1024 2x48x48")

        // 3x3x3 with padding asymmetry (causal: pD front only handled by caller)
        runCorrectness(B:1, C_in:64,  D:4,  H:16, W:16,  C_out:64,   kD:3, kH:3, kW:3, pD:1, pH:1, pW:1, label:"3x3x3 B1 C64 4x16x16")
    }

    // ── perf test ─────────────────────────────────────────────────────────────

    private func runPerf(
        B: Int, C_in: Int, D: Int, H: Int, W: Int,
        C_out: Int, kD: Int, kH: Int, kW: Int,
        pD: Int = 0, pH: Int = 0, pW: Int = 0,
        label: String, warmup: Int = 3, iters: Int = 10
    ) {
        let inputN  = B * C_in * D * H * W
        let weightN = C_out * C_in * kD * kH * kW

        let xBuf = randF16(inputN,  seed: 1)
        let wBuf = randF16(weightN, seed: 2)
        let bBuf = randF16(C_out,   seed: 3)
        let xT = Tensor(buffer: xBuf, shape: [B, C_in, D, H, W],        dtype: .float16)
        let wT = Tensor(buffer: wBuf, shape: [C_out, C_in, kD, kH, kW], dtype: .float16)
        let bT = Tensor(buffer: bBuf, shape: [C_out],                    dtype: .float16)

        // warmup
        for _ in 0..<warmup {
            _ = conv3d(xT, weight: wT, bias: bT, B:B, C_in:C_in, D:D, H:H, W:W,
                       C_out:C_out, kD:kD, kH:kH, kW:kW, pD:pD, pH:pH, pW:pW)
            _ = conv3d_implicit(xT, weight: wT, bias: bT, B:B, C_in:C_in, D:D, H:H, W:W,
                                C_out:C_out, kD:kD, kH:kH, kW:kW, pD:pD, pH:pH, pW:pW)
        }

        // time im2col
        var t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            _ = conv3d(xT, weight: wT, bias: bT, B:B, C_in:C_in, D:D, H:H, W:W,
                       C_out:C_out, kD:kD, kH:kH, kW:kW, pD:pD, pH:pH, pW:pW)
        }
        let msRef = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(iters)

        // time implicit
        t0 = CFAbsoluteTimeGetCurrent()
        for _ in 0..<iters {
            _ = conv3d_implicit(xT, weight: wT, bias: bT, B:B, C_in:C_in, D:D, H:H, W:W,
                                C_out:C_out, kD:kD, kH:kH, kW:kW, pD:pD, pH:pH, pW:pW)
        }
        let msNew = (CFAbsoluteTimeGetCurrent() - t0) * 1000.0 / Double(iters)

        let speedup = msRef / msNew
        print(String(format: "  %-40@  im2col=%6.1fms  implicit=%6.1fms  %.2fx",
                     label, msRef, msNew, speedup))
    }

    func testPerf() {
        print("\n── Conv3D perf: im2col vs implicit GEMM ─────────────────────────────")

        runPerf(B:1, C_in:128,  D:2,  H:48,  W:48,  C_out:1024, kD:3, kH:3, kW:3, pD:1, pH:1, pW:1, label:"conv_in   C128->1024 2x48x48")
        runPerf(B:1, C_in:1024, D:2,  H:48,  W:48,  C_out:1024, kD:3, kH:3, kW:3, pD:1, pH:1, pW:1, label:"midblock  C1024     2x48x48")
        runPerf(B:1, C_in:512,  D:4,  H:96,  W:96,  C_out:512,  kD:3, kH:3, kW:3, pD:1, pH:1, pW:1, label:"midblock  C512      4x96x96")
        runPerf(B:1, C_in:256,  D:8,  H:192, W:192, C_out:256,  kD:3, kH:3, kW:3, pD:1, pH:1, pW:1, label:"midblock  C256      8x192x192")
    }
}
