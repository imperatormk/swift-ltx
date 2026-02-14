// Element-wise Metal kernels for transformer ops.
// These are simple kernels compiled at runtime via makeLibrary(source:).
// The heavy stuff (GEMM, attention) comes from FlashAttention package.

import Metal

/// Compiled kernel cache.
public final class KernelCache: @unchecked Sendable {
    nonisolated(unsafe) public static let shared = KernelCache()

    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let library: MTLLibrary

    private init() {
        let ctx = MetalContext.shared
        do {
            self.library = try ctx.device.makeLibrary(source: KernelCache.metalSource, options: nil)
        } catch {
            fatalError("Metal shader compilation failed: \(error)")
        }
    }

    public func pipeline(_ name: String) -> MTLComputePipelineState {
        if let p = pipelines[name] { return p }
        let fn = library.makeFunction(name: name)!
        let p = try! MetalContext.shared.device.makeComputePipelineState(function: fn)
        pipelines[name] = p
        return p
    }

    static let metalSource = """
    #include <metal_stdlib>
    using namespace metal;

    // RMS Norm: out[i] = (x[i] / rms) * weight[i]
    // Parallel reduction across threads in threadgroup.
    // Grid: (rows) threadgroups, threadgroupSize threads each.
    // Input/output f16, weight f32, accumulator f32.
    kernel void rms_norm(
        device const half* x [[buffer(0)]],
        device const float* weight [[buffer(1)]],
        device half* out [[buffer(2)]],
        constant uint& dim [[buffer(3)]],
        constant float& eps [[buffer(4)]],
        uint tg_id [[threadgroup_position_in_grid]],
        uint tid_in_tg [[thread_index_in_threadgroup]],
        uint tg_size [[threads_per_threadgroup]])
    {
        uint row = tg_id;
        uint off = row * dim;

        // Parallel sum of squares
        float sum_sq = 0;
        for (uint i = tid_in_tg; i < dim; i += tg_size) {
            float v = float(x[off + i]);
            sum_sq += v * v;
        }

        // SIMD reduction
        sum_sq += simd_shuffle_xor(sum_sq, 1);
        sum_sq += simd_shuffle_xor(sum_sq, 2);
        sum_sq += simd_shuffle_xor(sum_sq, 4);
        sum_sq += simd_shuffle_xor(sum_sq, 8);
        sum_sq += simd_shuffle_xor(sum_sq, 16);

        threadgroup float shared[8];
        uint simd_lane = tid_in_tg % 32;
        uint simd_id = tid_in_tg / 32;
        if (simd_lane == 0) shared[simd_id] = sum_sq;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid_in_tg == 0) {
            float total_sq = 0;
            uint num_simds = (tg_size + 31) / 32;
            for (uint i = 0; i < num_simds; i++) total_sq += shared[i];
            shared[0] = rsqrt(total_sq / float(dim) + eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float rms = shared[0];

        // Apply norm + weight in parallel
        for (uint i = tid_in_tg; i < dim; i += tg_size) {
            out[off + i] = half(float(x[off + i]) * rms * weight[i]);
        }
    }

    // Fused SiLU + element-wise multiply: out = silu(a) * b
    // Saves a kernel launch and memory round-trip.
    kernel void silu_mul(
        device const half* a [[buffer(0)]],
        device const half* b [[buffer(1)]],
        device half* out [[buffer(2)]],
        uint tid [[thread_position_in_grid]])
    {
        float v = float(a[tid]);
        out[tid] = half((v / (1.0f + exp(-v))) * float(b[tid]));
    }

    // SiLU (Swish): out = x * sigmoid(x) = x / (1 + exp(-x))
    kernel void silu(
        device const half* x [[buffer(0)]],
        device half* out [[buffer(1)]],
        uint tid [[thread_position_in_grid]])
    {
        float v = float(x[tid]);
        out[tid] = half(v / (1.0f + exp(-v)));
    }

    // Element-wise multiply: out = a * b
    kernel void mul(
        device const half* a [[buffer(0)]],
        device const half* b [[buffer(1)]],
        device half* out [[buffer(2)]],
        uint tid [[thread_position_in_grid]])
    {
        out[tid] = a[tid] * b[tid];
    }

    // Fused residual add: out = a + b (same as add, kept for clarity)
    kernel void add(
        device const half* a [[buffer(0)]],
        device const half* b [[buffer(1)]],
        device half* out [[buffer(2)]],
        uint tid [[thread_position_in_grid]])
    {
        out[tid] = a[tid] + b[tid];
    }

    // Transpose [seqLen, nHeads, headDim] → [nHeads, seqLen, headDim]
    // Grid: (headDim, seqLen * nHeads)
    kernel void transpose_sh(
        device const half* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& seqLen [[buffer(2)]],
        constant uint& nHeads [[buffer(3)]],
        constant uint& headDim [[buffer(4)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint d = tid.x;
        uint idx = tid.y;
        uint s = idx / nHeads;
        uint h = idx % nHeads;
        dst[h * seqLen * headDim + s * headDim + d] = src[s * nHeads * headDim + h * headDim + d];
    }

    // Transpose [nHeads, seqLen, headDim] → [seqLen, nHeads, headDim]
    kernel void transpose_hs(
        device const half* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& seqLen [[buffer(2)]],
        constant uint& nHeads [[buffer(3)]],
        constant uint& headDim [[buffer(4)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint d = tid.x;
        uint idx = tid.y;
        uint h = idx / seqLen;
        uint s = idx % seqLen;
        dst[s * nHeads * headDim + h * headDim + d] = src[h * seqLen * headDim + s * headDim + d];
    }

    // RoPE (Rotary Position Embedding) with precomputed frequencies.
    // Uses half-dim split pairing: pair i rotates (x[i], x[i + D/2]).
    // Input/output f16, freqs f32, trig computed in f32.
    kernel void rope(
        device const half* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& headDim [[buffer(2)]],
        constant uint& startPos [[buffer(3)]],
        device const float* freqs [[buffer(4)]],
        constant uint& seqLen [[buffer(5)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint pair = tid.x;
        uint hs = tid.y;
        uint seq_offset = hs % seqLen;
        uint position = startPos + seq_offset;

        uint halfDim = headDim / 2;
        uint idx0 = hs * headDim + pair;
        uint idx1 = idx0 + halfDim;

        float freq = freqs[pair];
        float angle = float(position) * freq;
        float cos_a = cos(angle);
        float sin_a = sin(angle);

        float x0 = float(src[idx0]);
        float x1 = float(src[idx1]);
        dst[idx0] = half(x0 * cos_a - x1 * sin_a);
        dst[idx1] = half(x0 * sin_a + x1 * cos_a);
    }

    // Copy a contiguous slice from src to dst (uint4 = 16 bytes per thread).
    kernel void copy_slice(
        device const uint4* src [[buffer(0)]],
        device uint4* dst [[buffer(1)]],
        uint tid [[thread_position_in_grid]])
    {
        dst[tid] = src[tid];
    }

    // GPU zero-fill (uint4 = 16 bytes per thread for bandwidth).
    kernel void fill_zero(
        device uint4* dst [[buffer(0)]],
        uint tid [[thread_position_in_grid]])
    {
        dst[tid] = uint4(0);
    }

    // KV cache append: copy [numKVHeads, seqLen, headDim] into strided [numKVHeads, maxSeqLen, headDim]
    // Grid: (headDim, numKVHeads * seqLen) threads
    kernel void kv_append(
        device const half* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& headDim [[buffer(2)]],
        constant uint& seqLen [[buffer(3)]],
        constant uint& maxSeqLen [[buffer(4)]],
        constant uint& startPos [[buffer(5)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint d = tid.x;
        uint idx = tid.y;
        uint h = idx / seqLen;
        uint s = idx % seqLen;
        uint src_off = h * seqLen * headDim + s * headDim + d;
        uint dst_off = h * maxSeqLen * headDim + (startPos + s) * headDim + d;
        dst[dst_off] = src[src_off];
    }

    // KV cache compact: copy strided [numKVHeads, maxSeqLen, headDim] → contiguous [numKVHeads, seqLen, headDim]
    // Grid: (headDim, numKVHeads * seqLen) threads
    kernel void kv_compact(
        device const half* src [[buffer(0)]],
        device half* dst [[buffer(1)]],
        constant uint& headDim [[buffer(2)]],
        constant uint& seqLen [[buffer(3)]],
        constant uint& maxSeqLen [[buffer(4)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint d = tid.x;
        uint idx = tid.y;
        uint h = idx / seqLen;
        uint s = idx % seqLen;
        uint src_off = h * maxSeqLen * headDim + s * headDim + d;
        uint dst_off = h * seqLen * headDim + s * headDim + d;
        dst[dst_off] = src[src_off];
    }

    // Softmax over last dimension (for logits → probs).
    kernel void softmax_1d(
        device float* x [[buffer(0)]],
        constant uint& n [[buffer(1)]],
        uint tid [[thread_position_in_grid]])
    {
        if (tid != 0) return;

        float max_val = -INFINITY;
        for (uint i = 0; i < n; i++) max_val = max(max_val, x[i]);

        float sum = 0;
        for (uint i = 0; i < n; i++) {
            x[i] = exp(x[i] - max_val);
            sum += x[i];
        }
        for (uint i = 0; i < n; i++) x[i] /= sum;
    }

    // Batch embedding lookup: out[t * dim + i] = table[token_ids[t] * dim + i]
    // Table stays f32 (weight data), output f16.
    kernel void embedding_batch(
        device const float* table [[buffer(0)]],
        device half* out [[buffer(1)]],
        device const uint* token_ids [[buffer(2)]],
        constant uint& dim [[buffer(3)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint i = tid.x;
        uint t = tid.y;
        out[t * dim + i] = half(table[token_ids[t] * dim + i]);
    }

    // Batch quantized embedding lookup — output f16.
    kernel void embedding_q4_batch(
        device const uint* weight [[buffer(0)]],
        device const half* scales [[buffer(1)]],
        device const half* biases [[buffer(2)]],
        device half* out [[buffer(3)]],
        device const uint* token_ids [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        constant uint& group_size [[buffer(6)]],
        uint2 tid [[thread_position_in_grid]])
    {
        uint k_idx = tid.x;
        uint t = tid.y;
        uint token_id = token_ids[t];
        uint packed_k = K / 8;
        uint groups_per_row = K / group_size;

        uint pack_idx = k_idx / 8;
        uint sub_idx = k_idx % 8;
        uint group_idx = k_idx / group_size;

        uint packed_val = weight[token_id * packed_k + pack_idx];
        uint nibble = (packed_val >> (sub_idx * 4)) & 0xF;

        float scale = float(scales[token_id * groups_per_row + group_idx]);
        float bias = float(biases[token_id * groups_per_row + group_idx]);

        out[t * K + k_idx] = half(float(nibble) * scale + bias);
    }

    // Embedding lookup — table f32, output f16.
    kernel void embedding(
        device const float* table [[buffer(0)]],
        device half* out [[buffer(1)]],
        constant uint& token_id [[buffer(2)]],
        constant uint& dim [[buffer(3)]],
        uint tid [[thread_position_in_grid]])
    {
        out[tid] = half(table[token_id * dim + tid]);
    }

    // Quantized embedding lookup (4-bit) — output f16.
    kernel void embedding_q4(
        device const uint* weight [[buffer(0)]],
        device const half* scales [[buffer(1)]],
        device const half* biases [[buffer(2)]],
        device half* out [[buffer(3)]],
        constant uint& token_id [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        constant uint& group_size [[buffer(6)]],
        uint tid [[thread_position_in_grid]])
    {
        uint packed_k = K / 8;
        uint groups_per_row = K / group_size;

        uint pack_idx = tid / 8;
        uint sub_idx = tid % 8;
        uint group_idx = tid / group_size;

        uint packed_val = weight[token_id * packed_k + pack_idx];
        uint nibble = (packed_val >> (sub_idx * 4)) & 0xF;

        float scale = float(scales[token_id * groups_per_row + group_idx]);
        float bias = float(biases[token_id * groups_per_row + group_idx]);

        out[tid] = half(float(nibble) * scale + bias);
    }

    // Quantized matvec: out = x @ W^T where W is 4-bit quantized.
    // Each threadgroup computes one output row (one n).
    // Threads split work across K dimension with vectorized uint32 loads.
    // Input x is f16, accumulator f32, output f16.
    kernel void matmul_q4(
        device const half* x [[buffer(0)]],
        device const uint* weight [[buffer(1)]],
        device const half* scales [[buffer(2)]],
        device const half* biases [[buffer(3)]],
        device half* out [[buffer(4)]],
        constant uint& K [[buffer(5)]],
        constant uint& group_size [[buffer(6)]],
        constant uint& N [[buffer(7)]],
        uint tg_id [[threadgroup_position_in_grid]],
        uint tid_in_tg [[thread_index_in_threadgroup]],
        uint tg_size [[threads_per_threadgroup]])
    {
        uint n = tg_id % N;
        uint m = tg_id / N;
        uint packed_k = K / 8;
        uint groups_per_row = K / group_size;
        uint packs_per_group = group_size / 8;

        // Vectorized: each thread processes 4 packed uint32s (32 weights) per iteration
        float sum = 0.0f;
        device const uint4* w_vec = (device const uint4*)(weight + n * packed_k);
        device const half4* x_base = (device const half4*)(x + m * K);

        uint total_vec4 = packed_k / 4;
        for (uint vi = tid_in_tg; vi < total_vec4; vi += tg_size) {
            uint4 packed4 = w_vec[vi];
            uint p_base = vi * 4;

            // Process 4 packed uint32s = 32 weights
            uint packs[4] = {packed4.x, packed4.y, packed4.z, packed4.w};
            for (uint pi = 0; pi < 4; pi++) {
                uint p = p_base + pi;
                uint g = p / packs_per_group;
                float scale = float(scales[n * groups_per_row + g]);
                float bias = float(biases[n * groups_per_row + g]);

                uint packed_val = packs[pi];
                uint k_off = p * 8;

                // Unrolled 8-nibble extraction with half4 x loads
                half4 xv0 = x_base[k_off / 4];
                half4 xv1 = x_base[k_off / 4 + 1];

                sum += (float((packed_val      ) & 0xF) * scale + bias) * float(xv0.x);
                sum += (float((packed_val >>  4) & 0xF) * scale + bias) * float(xv0.y);
                sum += (float((packed_val >>  8) & 0xF) * scale + bias) * float(xv0.z);
                sum += (float((packed_val >> 12) & 0xF) * scale + bias) * float(xv0.w);
                sum += (float((packed_val >> 16) & 0xF) * scale + bias) * float(xv1.x);
                sum += (float((packed_val >> 20) & 0xF) * scale + bias) * float(xv1.y);
                sum += (float((packed_val >> 24) & 0xF) * scale + bias) * float(xv1.z);
                sum += (float((packed_val >> 28) & 0xF) * scale + bias) * float(xv1.w);
            }
        }

        // Handle remainder (packed_k not multiple of 4)
        uint remainder_start = total_vec4 * 4;
        for (uint p = remainder_start + tid_in_tg; p < packed_k; p += tg_size) {
            uint g = p / packs_per_group;
            float scale = float(scales[n * groups_per_row + g]);
            float bias = float(biases[n * groups_per_row + g]);
            uint packed_val = weight[n * packed_k + p];
            uint k_off = p * 8;
            for (uint s = 0; s < 8; s++) {
                uint nibble = (packed_val >> (s * 4)) & 0xF;
                sum += (float(nibble) * scale + bias) * float(x[m * K + k_off + s]);
            }
        }

        // SIMD reduction
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        sum += simd_shuffle_xor(sum, 4);
        sum += simd_shuffle_xor(sum, 8);
        sum += simd_shuffle_xor(sum, 16);

        // Cross-SIMD reduction
        threadgroup float shared[8];
        uint simd_lane = tid_in_tg % 32;
        uint simd_id = tid_in_tg / 32;
        if (simd_lane == 0) shared[simd_id] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tid_in_tg == 0) {
            float total = 0;
            uint num_simds = (tg_size + 31) / 32;
            for (uint i = 0; i < num_simds; i++) total += shared[i];
            out[m * N + n] = half(total);
        }
    }

    // Naive GPU attention with causal mask — parallelized across D dimension.
    // Q=[nHeads, R, D], K=[nKVHeads, C, D], V=[nKVHeads, C, D] → O=[nHeads, R, D]
    // Grid: (nHeads, R) threadgroups, D threads each.
    // params: [nHeads, nKVHeads, R, C, D, startPos]
    // Input Q/K/V f16, accumulator f32, output O f16.
    kernel void naive_attention(
        device const half* Q [[buffer(0)]],
        device const half* K [[buffer(1)]],
        device const half* V [[buffer(2)]],
        device half* O [[buffer(3)]],
        constant uint* params [[buffer(4)]],
        uint2 tg_id [[threadgroup_position_in_grid]],
        uint tid_in_tg [[thread_index_in_threadgroup]],
        uint2 tg_size2 [[threads_per_threadgroup]])
    {
        uint tg_size = tg_size2.x;
        uint h = tg_id.x;
        uint r = tg_id.y;
        uint nHeads = params[0];
        uint nKVHeads = params[1];
        uint R = params[2];
        uint C = params[3];
        uint D = params[4];
        uint startPos = params[5];
        uint kvRepeat = nHeads / nKVHeads;
        uint kvh = h / kvRepeat;

        uint causal_limit = min(startPos + r + 1, C);
        float scale = rsqrt(float(D));

        uint q_base = h * R * D + r * D;
        uint kv_base = kvh * C * D;

        // Shared memory for reductions (must be at function scope)
        threadgroup float tg_reduce[8];
        threadgroup float tg_broadcast;

        // Each thread handles a subset of D dimensions
        float q_local[4];
        uint d_per_thread = (D + tg_size - 1) / tg_size;
        uint d_start = tid_in_tg * d_per_thread;
        uint d_end = min(d_start + d_per_thread, D);
        uint d_count = (d_start < D) ? (d_end - d_start) : 0;
        for (uint i = 0; i < d_count && i < 4; i++) {
            q_local[i] = float(Q[q_base + d_start + i]);
        }

        uint simd_lane = tid_in_tg % 32;
        uint simd_id = tid_in_tg / 32;

        // Helper: reduce partial_dot across all threads, broadcast result
        // After this, `dot` is the same value in ALL threads.
        #define REDUCE_DOT(partial_dot, result) \
            partial_dot += simd_shuffle_xor(partial_dot, 1); \
            partial_dot += simd_shuffle_xor(partial_dot, 2); \
            partial_dot += simd_shuffle_xor(partial_dot, 4); \
            partial_dot += simd_shuffle_xor(partial_dot, 8); \
            partial_dot += simd_shuffle_xor(partial_dot, 16); \
            if (simd_lane == 0) tg_reduce[simd_id] = partial_dot; \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            if (tid_in_tg == 0) { \
                float _s = 0; \
                uint _n = (tg_size + 31) / 32; \
                for (uint _i = 0; _i < _n; _i++) _s += tg_reduce[_i]; \
                tg_broadcast = _s; \
            } \
            threadgroup_barrier(mem_flags::mem_threadgroup); \
            float result = tg_broadcast;

        // Pass 1: find max score
        float max_s = -INFINITY;
        for (uint s = 0; s < causal_limit; s++) {
            float partial_dot = 0;
            for (uint i = 0; i < d_count && i < 4; i++) {
                partial_dot += q_local[i] * float(K[kv_base + s * D + d_start + i]);
            }
            REDUCE_DOT(partial_dot, dot)
            max_s = max(max_s, dot * scale);
        }

        // Pass 2: softmax + weighted V accumulation
        float sum_e = 0;
        float o_acc[4] = {0, 0, 0, 0};
        for (uint s = 0; s < causal_limit; s++) {
            float partial_dot = 0;
            for (uint i = 0; i < d_count && i < 4; i++) {
                partial_dot += q_local[i] * float(K[kv_base + s * D + d_start + i]);
            }
            REDUCE_DOT(partial_dot, dot2)
            float p = exp(dot2 * scale - max_s);
            sum_e += p;
            for (uint i = 0; i < d_count && i < 4; i++) {
                o_acc[i] += p * float(V[kv_base + s * D + d_start + i]);
            }
        }

        #undef REDUCE_DOT

        // Write output
        float inv_sum = 1.0f / sum_e;
        uint o_base = h * R * D + r * D;
        for (uint i = 0; i < d_count && i < 4; i++) {
            O[o_base + d_start + i] = half(o_acc[i] * inv_sum);
        }
    }
    """
}
