// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwiftLLM",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "SwiftLLM", targets: ["SwiftLLM"]),
    ],
    dependencies: [
        .package(path: "../mps-flash-attention/metal-flash-attention"),  // FlashAttention kernels (includes MetalASM)
    ],
    targets: [
        .target(
            name: "SwiftLLM",
            dependencies: [
                .product(name: "FlashAttention", package: "metal-flash-attention"),
            ]
        ),
        .testTarget(
            name: "SwiftLLMTests",
            dependencies: [
                "SwiftLLM",
                .product(name: "FlashAttention", package: "metal-flash-attention"),
            ]
        ),
    ]
)
