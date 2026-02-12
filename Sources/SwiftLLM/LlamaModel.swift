// Llama transformer model — pure Swift + Metal.
// Loads weights from safetensors, runs inference on GPU.

import Metal
import FlashAttention
import MetalASM

public class LlamaModel: @unchecked Sendable {
    public let config: ModelConfig
    private let weights: ModelWeights
    private let cache: KVCache
    public var useFlashAttention: Bool = true

    /// Cached dynamic-RC decode kernel (compiled once, reused for all C values).
    private var dynamicDecodeKernel: AttentionKernel?
    private var dynamicDecodePipeline: MTLComputePipelineState?

    public init(directory: URL) throws {
        self.config = try ModelConfig(from: directory.appendingPathComponent("config.json"))
        let files = try loadSafeTensors(from: directory)
        self.weights = try ModelWeights(config: config, files: files)
        self.cache = KVCache(
            numLayers: config.numHiddenLayers,
            numKVHeads: config.numKeyValueHeads,
            headDim: config.headDim)
    }

    /// Compile the dynamic-RC decode kernel (R=1, any C). Called lazily on first decode.
    private func ensureDynamicDecodeKernel() {
        guard dynamicDecodeKernel == nil else { return }
        let headDim = config.headDim

        // Use a nominal C for block dimension selection; actual C comes from batch_params.
        var attDesc = AttentionDescriptor()
        attDesc.lowPrecisionInputs = false
        attDesc.lowPrecisionIntermediates = false
        attDesc.matrixDimensions = (row: 1, column: 4096, head: UInt16(headDim))
        attDesc.transposeState = (Q: false, K: false, V: false, O: false)
        attDesc.causal = false

        let kernelDesc = attDesc.kernelDescriptor(type: .forward)
        let kernel = AttentionKernel(descriptor: kernelDesc)

        var monoDesc = AttentionKernel.MonolithicDescriptor()
        monoDesc.R = 1
        monoDesc.C = 4096
        monoDesc.dynamicRC = true
        let d = UInt32(headDim)
        monoDesc.leadingDimensions[.Q] = d
        monoDesc.leadingDimensions[.K] = d
        monoDesc.leadingDimensions[.V] = d
        monoDesc.leadingDimensions[.O] = d

        let ir = kernel.createSource(descriptor: monoDesc)
        #if os(macOS)
        let metallibData = try! MetalASM.assemble(ir: ir, platform: .macOS(version: 26))
        #elseif os(iOS)
        let metallibData = try! MetalASM.assemble(ir: ir, platform: .iOS(version: 26))
        #endif

        let device = MetalContext.shared.device
        let dispatchData = metallibData.withUnsafeBytes { DispatchData(bytes: $0) }
        let library = try! device.makeLibrary(data: dispatchData)
        let function = library.makeFunction(name: "attention")!
        let pipeline = try! device.makeComputePipelineState(function: function)

        self.dynamicDecodeKernel = kernel
        self.dynamicDecodePipeline = pipeline
    }

    /// Generate tokens autoregressively.
    public func generate(prompt: [Int], maxTokens: Int = 256, temperature: Float = 0.7, topP: Float = 0.9, onToken: (Int, String) -> Bool) {
        cache.reset()

        // Prefill: process all prompt tokens in one forward pass
        let lastLogits = prefill(tokenIds: prompt)
        cache.incrementPosition(by: prompt.count)

        // Decode
        var nextToken = temperature > 0 ? sample(lastLogits, temperature: temperature, topP: topP) : argmax(lastLogits)
        for _ in 0..<maxTokens {
            if !onToken(nextToken, "") { break }
            let logits = forward(tokenId: nextToken, position: cache.currentLen)
            cache.incrementPosition()
            nextToken = temperature > 0 ? sample(logits, temperature: temperature, topP: topP) : argmax(logits)
        }
    }

    /// Prefill: process all prompt tokens in one batched forward pass.
    /// Returns logits for the LAST token only (for next-token prediction).
    private func prefill(tokenIds: [Int]) -> Tensor {
        MetalContext.shared.bufferPool.reset()
        let seqLen = tokenIds.count
        let hidden = config.hiddenSize

        // Batch embedding: [seqLen, hidden]
        var x: Tensor
        switch weights.embedTokens {
        case .float(let table):
            x = embeddingBatch(table: table, tokenIds: tokenIds, dim: hidden)
        case .quantized(let weight, let scales, let biases, let K, let groupSize):
            x = embeddingQ4Batch(weight: weight, scales: scales, biases: biases, tokenIds: tokenIds, K: K, groupSize: groupSize)
        }

        let pool = MetalContext.shared.bufferPool
        for layer in 0..<config.numHiddenLayers {
            x = transformerBlockBatch(x, layer: layer, seqLen: seqLen, startPos: 0)
            pool.reset(keeping: [x.buffer])
        }

        // Final norm on full [seqLen, hidden]
        x = rmsNorm(x, weight: weights.normWeight, eps: config.rmsNormEps, dim: hidden)

        // LM head on last token only: extract [1, hidden] from [seqLen, hidden]
        let lastRow = Tensor(buffer: x.buffer, shape: [1, hidden],
                            offset: (seqLen - 1) * hidden * 4)
        return linear(lastRow, weights.lmHead, M: 1, K: hidden, N: config.vocabSize, fast: useFlashAttention)
    }

    /// Single-token forward pass (for decode after prefill).
    private func forward(tokenId: Int, position: Int) -> Tensor {
        MetalContext.shared.bufferPool.reset()
        var x: Tensor
        switch weights.embedTokens {
        case .float(let table):
            x = embedding(table: table, tokenId: tokenId, dim: config.hiddenSize)
        case .quantized(let weight, let scales, let biases, let K, let groupSize):
            x = embeddingQ4(weight: weight, scales: scales, biases: biases, tokenId: tokenId, K: K, groupSize: groupSize)
        }

        let pool = MetalContext.shared.bufferPool
        for layer in 0..<config.numHiddenLayers {
            x = transformerBlock(x, layer: layer, position: position)
            pool.reset(keeping: [x.buffer])
        }

        x = rmsNorm(x, weight: weights.normWeight, eps: config.rmsNormEps, dim: config.hiddenSize)
        x = linear(x, weights.lmHead, M: 1, K: config.hiddenSize, N: config.vocabSize, fast: useFlashAttention)
        return x
    }

    // MARK: - Batched transformer block (prefill)

    private func transformerBlockBatch(_ x: Tensor, layer: Int, seqLen: Int, startPos: Int) -> Tensor {
        let w = weights.layers[layer]
        let hidden = config.hiddenSize
        let nHeads = config.numAttentionHeads
        let nKVHeads = config.numKeyValueHeads
        let headDim = config.headDim

        // Self-attention
        let normed = rmsNorm(x, weight: w.inputNormWeight, eps: config.rmsNormEps, dim: hidden)

        // QKV projections: [seqLen, hidden] → [seqLen, nHeads*headDim]
        var q = linear(normed, w.qProj, M: seqLen, K: hidden, N: nHeads * headDim, fast: useFlashAttention)
        var k = linear(normed, w.kProj, M: seqLen, K: hidden, N: nKVHeads * headDim, fast: useFlashAttention)
        let v = linear(normed, w.vProj, M: seqLen, K: hidden, N: nKVHeads * headDim, fast: useFlashAttention)

        // Transpose [seqLen, nHeads*headDim] → [nHeads, seqLen, headDim] for RoPE
        q = transposeSH(q, seqLen: seqLen, nHeads: nHeads, headDim: headDim)
        k = transposeSH(k, seqLen: seqLen, nHeads: nKVHeads, headDim: headDim)

        q = rope(q, headDim: headDim, seqLen: seqLen, startPos: startPos, theta: config.ropeTheta)
        k = rope(k, headDim: headDim, seqLen: seqLen, startPos: startPos, theta: config.ropeTheta)

        // Update KV cache with all tokens
        let kForCache = k
        let vReshaped = transposeSH(v, seqLen: seqLen, nHeads: nKVHeads, headDim: headDim)
        cache.append(layer: layer, k: kForCache, v: vReshaped, seqLen: seqLen)

        // For prefill: Q and K/V are the same tokens, C = seqLen
        let totalSeq = startPos + seqLen
        let attnOut: Tensor
        if useFlashAttention {
            attnOut = flashAttention(
                q: q, k: cache.keys(layer: layer, seqLen: totalSeq),
                v: cache.values(layer: layer, seqLen: totalSeq),
                R: seqLen, nHeads: nHeads, nKVHeads: nKVHeads,
                seqLen: totalSeq, headDim: headDim)
        } else {
            attnOut = naiveAttention(
                q: q, k: cache.keys(layer: layer, seqLen: totalSeq),
                v: cache.values(layer: layer, seqLen: totalSeq),
                R: seqLen, nHeads: nHeads, nKVHeads: nKVHeads,
                seqLen: totalSeq, headDim: headDim, startPos: startPos)
        }

        // Transpose [nHeads, seqLen, headDim] → [seqLen, nHeads*headDim] for O projection
        let attnFlat = transposeHS(attnOut, seqLen: seqLen, nHeads: nHeads, headDim: headDim)
        let attnFlatReshaped = Tensor(buffer: attnFlat.buffer, shape: [seqLen, nHeads * headDim])
        let attnProj = linear(attnFlatReshaped, w.oProj, M: seqLen, K: nHeads * headDim, N: hidden, fast: useFlashAttention)

        let afterAttn = elemAdd(x, attnProj)

        // FFN (SwiGLU)
        let normed2 = rmsNorm(afterAttn, weight: w.postNormWeight, eps: config.rmsNormEps, dim: hidden)
        let gate = linear(normed2, w.gateProj, M: seqLen, K: hidden, N: config.intermediateSize, fast: useFlashAttention)
        let up = linear(normed2, w.upProj, M: seqLen, K: hidden, N: config.intermediateSize, fast: useFlashAttention)
        let activated = siluMul(gate, up)
        let down = linear(activated, w.downProj, M: seqLen, K: config.intermediateSize, N: hidden, fast: useFlashAttention)

        return elemAdd(afterAttn, down)
    }

    // MARK: - Single-token transformer block (decode)

    private func transformerBlock(_ x: Tensor, layer: Int, position: Int) -> Tensor {
        let w = weights.layers[layer]

        let normed = rmsNorm(x, weight: w.inputNormWeight, eps: config.rmsNormEps, dim: config.hiddenSize)

        var q = linear(normed, w.qProj, M: 1, K: config.hiddenSize, N: config.numAttentionHeads * config.headDim, fast: useFlashAttention)
        var k = linear(normed, w.kProj, M: 1, K: config.hiddenSize, N: config.numKeyValueHeads * config.headDim, fast: useFlashAttention)
        let v = linear(normed, w.vProj, M: 1, K: config.hiddenSize, N: config.numKeyValueHeads * config.headDim, fast: useFlashAttention)

        q = Tensor(buffer: q.buffer, shape: [config.numAttentionHeads, 1, config.headDim])
        k = Tensor(buffer: k.buffer, shape: [config.numKeyValueHeads, 1, config.headDim])

        q = rope(q, headDim: config.headDim, seqLen: 1, startPos: position, theta: config.ropeTheta)
        k = rope(k, headDim: config.headDim, seqLen: 1, startPos: position, theta: config.ropeTheta)

        let kForCache = Tensor(buffer: k.buffer, shape: [config.numKeyValueHeads, 1, config.headDim])
        let vForCache = Tensor(buffer: v.buffer, shape: [config.numKeyValueHeads, 1, config.headDim])
        cache.append(layer: layer, k: kForCache, v: vForCache)

        let seqLen = cache.currentLen + 1
        let attnOut: Tensor
        if useFlashAttention {
            attnOut = flashAttention(
                q: q, k: cache.keys(layer: layer, seqLen: seqLen), v: cache.values(layer: layer, seqLen: seqLen),
                R: 1, nHeads: config.numAttentionHeads, nKVHeads: config.numKeyValueHeads,
                seqLen: seqLen, headDim: config.headDim)
        } else {
            attnOut = naiveAttention(
                q: q, k: cache.keys(layer: layer, seqLen: seqLen), v: cache.values(layer: layer, seqLen: seqLen),
                R: 1, nHeads: config.numAttentionHeads, nKVHeads: config.numKeyValueHeads,
                seqLen: seqLen, headDim: config.headDim, startPos: seqLen - 1)
        }

        let attnFlat = Tensor(buffer: attnOut.buffer, shape: [1, config.numAttentionHeads * config.headDim])
        let attnProj = linear(attnFlat, w.oProj, M: 1, K: config.numAttentionHeads * config.headDim, N: config.hiddenSize, fast: useFlashAttention)

        let afterAttn = elemAdd(x, attnProj)

        let normed2 = rmsNorm(afterAttn, weight: w.postNormWeight, eps: config.rmsNormEps, dim: config.hiddenSize)
        let gate = linear(normed2, w.gateProj, M: 1, K: config.hiddenSize, N: config.intermediateSize, fast: useFlashAttention)
        let up = linear(normed2, w.upProj, M: 1, K: config.hiddenSize, N: config.intermediateSize, fast: useFlashAttention)
        let activated = siluMul(gate, up)
        let down = linear(activated, w.downProj, M: 1, K: config.intermediateSize, N: config.hiddenSize, fast: useFlashAttention)

        return elemAdd(afterAttn, down)
    }

    // MARK: - Attention dispatch

    private func flashAttention(
        q: Tensor, k: Tensor, v: Tensor,
        R: Int, nHeads: Int, nKVHeads: Int, seqLen: Int, headDim: Int
    ) -> Tensor {
        let kernel: AttentionKernel
        let pipeline: MTLComputePipelineState

        if R == 1 {
            // Decode: use dynamic-RC kernel (compiled once, reused for all C)
            ensureDynamicDecodeKernel()
            kernel = dynamicDecodeKernel!
            pipeline = dynamicDecodePipeline!
        } else {
            // Prefill: use static kernel (R/C baked, causal)
            var attDesc = AttentionDescriptor()
            attDesc.lowPrecisionInputs = false
            attDesc.lowPrecisionIntermediates = false
            attDesc.matrixDimensions = (row: UInt32(R), column: UInt32(seqLen), head: UInt16(headDim))
            attDesc.transposeState = (Q: false, K: false, V: false, O: false)
            attDesc.causal = true
            (kernel, pipeline) = AttentionKernel.pipeline(for: attDesc, type: .forward)
        }

        let pool = MetalContext.shared.bufferPool
        let oBytes = nHeads * R * headDim * 4
        let lBytes = nHeads * R * 4
        let bufO = pool.get(length: oBytes); memset(bufO.contents(), 0, oBytes)
        let bufL = pool.get(length: lBytes); memset(bufL.contents(), 0, lBytes)
        let dummy = pool.get(length: 4); memset(dummy.contents(), 0, 4)

        let batchParams = AttentionKernel.createBatchedParamsBuffer(
            numHeads: UInt32(nHeads),
            numKVHeads: UInt32(nKVHeads),
            R: UInt32(R), C: UInt32(seqLen), D: UInt32(headDim))

        MetalContext.shared.run { enc in
            enc.setBuffer(q.buffer, offset: 0, index: 0)
            enc.setBuffer(k.buffer, offset: 0, index: 1)
            enc.setBuffer(v.buffer, offset: 0, index: 2)
            enc.setBuffer(bufO, offset: 0, index: 3)
            enc.setBuffer(bufL, offset: 0, index: 4)
            enc.setBuffer(dummy, offset: 0, index: 5)
            enc.setBuffer(dummy, offset: 0, index: 6)
            enc.setBuffer(dummy, offset: 0, index: 7)
            enc.setBuffer(dummy, offset: 0, index: 8)
            enc.setBuffer(dummy, offset: 0, index: 9)

            AttentionKernel.dispatch(
                encoder: enc,
                kernel: kernel,
                pipeline: pipeline,
                batchedParams: batchParams,
                parallelizationDimension: R,
                numHeads: nHeads,
                batchSize: 1)
        }

        return Tensor(buffer: bufO, shape: [nHeads, R, headDim])
    }

    private func naiveAttention(
        q: Tensor, k: Tensor, v: Tensor,
        R: Int, nHeads: Int, nKVHeads: Int, seqLen: Int, headDim: Int, startPos: Int
    ) -> Tensor {
        let bufO = MetalContext.shared.bufferPool.get(length: nHeads * R * headDim * 4)
        var params: [UInt32] = [UInt32(nHeads), UInt32(nKVHeads), UInt32(R), UInt32(seqLen), UInt32(headDim), UInt32(startPos)]
        let paramBuf = MetalContext.shared.makePooledBuffer(&params, length: params.count * 4)
        let pipe = KernelCache.shared.pipeline("naive_attention")

        MetalContext.shared.run { enc in
            enc.setComputePipelineState(pipe)
            enc.setBuffer(q.buffer, offset: 0, index: 0)
            enc.setBuffer(k.buffer, offset: 0, index: 1)
            enc.setBuffer(v.buffer, offset: 0, index: 2)
            enc.setBuffer(bufO, offset: 0, index: 3)
            enc.setBuffer(paramBuf, offset: 0, index: 4)
            let tgSize = min(256, headDim)
            enc.dispatchThreadgroups(MTLSize(width: nHeads, height: R, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        }
        return Tensor(buffer: bufO, shape: [nHeads, R, headDim])
    }
}

// MARK: - CPU matmul fallback (for projections — will replace with GEMM kernel)

/// Simple CPU matmul: C = A @ B^T, where A is [M, K] and B is [N, K] (row-major).
/// Output is [M, N].
func matmulCPU(_ a: Tensor, _ b: Tensor, M: Int, K: Int, N: Int) -> Tensor {
    let aPtr = a.buffer.contents().assumingMemoryBound(to: Float.self)
    let bPtr = b.buffer.contents().assumingMemoryBound(to: Float.self)

    var result = [Float](repeating: 0, count: M * N)
    for m in 0..<M {
        for n in 0..<N {
            var sum: Float = 0
            for k in 0..<K {
                sum += aPtr[m * K + k] * bPtr[n * K + k]
            }
            result[m * N + n] = sum
        }
    }
    return Tensor(result, shape: [M, N])
}
