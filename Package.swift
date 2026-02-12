// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "SwiftLLM",
    platforms: [.macOS(.v14), .iOS(.v17)],
    products: [
        .library(name: "SwiftLLM", targets: ["SwiftLLM"]),
        .executable(name: "swift-llm-cli", targets: ["CLI"]),
    ],
    dependencies: [
        .package(path: "../metal-flash-attention-upstream"),  // FlashAttention kernels (includes MetalASM)
    ],
    targets: [
        .target(
            name: "SwiftLLM",
            dependencies: [
                .product(name: "FlashAttention", package: "metal-flash-attention-upstream"),
            ]
        ),
        .executableTarget(
            name: "CLI",
            dependencies: ["SwiftLLM"]
        ),
        .executableTarget(
            name: "SwiftLLMApp",
            dependencies: ["SwiftLLM"],
            path: "App"
        ),
        .testTarget(
            name: "SwiftLLMTests",
            dependencies: ["SwiftLLM"]
        ),
    ]
)
