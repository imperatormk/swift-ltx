# SwiftLLM

Minimal Swift LLM inference on Apple GPU. Runs Llama-family models with 4-bit quantization (MLX format) on macOS and iOS.

## Features

- **Flash Attention** via [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) with monolithic IR kernels
- **Quantized GEMM** — 4-bit dequant directly to registers, assembled in-process via [MetalASM](https://github.com/mpsops/MetalASM)
- **Optimized naive kernels** — parallel RMS norm, vectorized matvec, fused SiLU+multiply
- **macOS + iOS** — universal Xcode project, works on Mac and iPhone

## Performance (M1 Pro, Llama 3.2 3B 4-bit, 750 token prompt)

| Mode | TTFT | Decode |
|------|------|--------|
| Flash Attention + Monolithic GEMM | **0.48s** | **6.0 tok/s** |
| Naive Attention + Basic Matvec | 1.0s | 4.8 tok/s |

## Setup

1. Clone with the [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) dependency next to this repo
2. Open `SwiftLLMDemo.xcodeproj` in Xcode
3. Set the model path to a HuggingFace directory containing `config.json`, `tokenizer.json`, and `*.safetensors` (MLX 4-bit quantized)
4. Build & Run

## Model

Tested with [mlx-community/Llama-3.2-3B-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.2-3B-Instruct-4bit). Any Llama-architecture model in MLX safetensors format should work.

## Architecture

```
Sources/SwiftLLM/     — inference library
  LlamaModel.swift    — model definition, prefill/decode loops
  Ops.swift           — Metal dispatch (matmul, attention, norm, etc.)
  Kernels.swift       — Metal shader source (naive kernels)
  ModelWeights.swift   — safetensors loader with 4-bit support
  Tokenizer.swift     — BPE tokenizer

SwiftLLMDemo/         — SwiftUI app (macOS + iOS)
```

## Dependencies

- [metal-flash-attention](https://github.com/philipturner/metal-flash-attention) — flash attention + monolithic GEMM kernels
- [MetalASM](https://github.com/mpsops/MetalASM) — in-process LLVM IR → metallib assembler
