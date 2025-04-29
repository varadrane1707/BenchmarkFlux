# Flux Text To Image Fast Inference Experimentation

This repository contains benchmarking results and analysis of various optimization techniques for the FLUX.1-dev model inference on NVIDIA A100 and H100 GPUs.

## Overview

The experimentation focuses on several key optimization strategies:
1. Data Type Optimizations (FP16 vs BF16)
2. Attention Mechanism Optimizations (FlashAttention2)
3. Attention Caching Strategies (TeaCache and FirstBlockCache)
4. Quantization (TorchAO FP8 and INT8)

## Benchmark Results

### A100 GPU Performance (1024x1024, 28 steps)

| Optimization Technique | Average Time (s) | Min Time (s) | P50 Time (s) | P90 Time (s) | P99 Time (s) |
|----------------------|------------------|--------------|--------------|--------------|--------------|
| Baseline (FP16)      | 16.13           | 16.01        | 16.14        | 16.18        | 16.19        |
| BF16                 | 15.65           | 15.61        | 15.66        | 15.68        | 15.69        |
| FlashAttention2      | 14.29           | 14.23        | 14.30        | 14.32        | 14.34        |
| TeaCache (0.2)       | 10.96           | 10.39        | 10.83        | 11.56        | 11.97        |
| TeaCache (0.4)       | 7.01            | 6.36         | 7.33         | 7.43         | 7.44         |
| TeaCache (0.6)       | 5.23            | 4.85         | 5.36         | 5.40         | 5.40         |
| FBCache (0.15)       | 5.09            | 4.99         | 5.04         | 5.16         | 5.53         |
| FBCache (0.2)        | 5.17            | 4.97         | 5.08         | 5.33         | 6.12         |
| FBCache (0.3)        | 4.09            | 4.03         | 4.07         | 4.12         | 4.34         |
| FBCache (0.35)       | 3.61            | 3.55         | 3.59         | 3.64         | 3.87         |
| FBCache (0.4)        | 3.31            | 3.04         | 3.13         | 3.62         | 3.82         |
| TeaCache+FA2 (0.6)   | 5.27            | 4.87         | 5.34         | 5.44         | 5.51         |
| FBCache+FA2 (0.2)    | 5.04            | 4.96         | 5.05         | 5.09         | 5.10         |

### H100 GPU Performance (1024x1024, 28 steps)

| Optimization Technique | Average Time (s) | Min Time (s) | P50 Time (s) | P90 Time (s) | P99 Time (s) |
|----------------------|------------------|--------------|--------------|--------------|--------------|
| Baseline (FP16)      | 7.23            | 7.18         | 7.18         | 7.19         | 7.66         |
| BF16                 | 7.11            | 7.09         | 7.09         | 7.10         | 7.24         |
| FlashAttention2      | 7.18            | 7.16         | 7.18         | 7.19         | 7.20         |
| TeaCache (0.2)       | 5.51            | 5.25         | 5.48         | 5.73         | 5.74         |
| TeaCache (0.4)       | 3.48            | 3.19         | 3.44         | 3.71         | 3.73         |
| TeaCache (0.6)       | 2.65            | 2.44         | 2.69         | 2.73         | 2.74         |
| FBCache (0.15)       | 3.21            | 3.03         | 3.30         | 3.32         | 3.68         |
| FBCache (0.2)        | 2.56            | 2.53         | 2.57         | 2.57         | 2.60         |
| FBCache (0.3)        | 2.05            | 2.09         | 2.10         | 2.12         | 2.30         |
| FBCache (0.35)       | 1.87            | 1.84         | 1.84         | 1.87         | 2.34         |
| TeaCache+FA2 (0.6)   | 2.63            | 2.45         | 2.70         | 2.73         | 2.73         |
| FBCache+FA2 (0.2)    | 3.20            | 3.02         | 3.27         | 3.28         | 3.28         |
| TorchAO FP8          | 7.28            | 7.19         | 7.20         | 7.25         | 8.05         |

### Advanced Configurations on H100

| Configuration | Average Time (s) | Min Time (s) | P50 Time (s) | P90 Time (s) | P99 Time (s) |
|--------------|------------------|--------------|--------------|--------------|--------------|
| BF16+FA2+TeaCache (0.6) | 2.72 | 2.41 | 2.66 | 2.72 | 4.39 |
| BF16+FA2+TeaCache (0.6)+TorchAO FP8 | 2.63 | 2.41 | 2.66 | 2.69 | 3.48 |
| FBCache (0.3) | 2.11 | 2.05 | 2.09 | 2.11 | 2.39 |
| BF16+FA2+FBCache (0.3) | 2.06 | 2.02 | 2.06 | 2.07 | 2.13 |
| BF16+FA2+FBCache (0.3)+TorchAO FP8 | 2.04 | 2.02 | 2.03 | 2.07 | 2.08 |
| BF16+FA2+FBCache (0.3)+TorchAO INT8 | 2.04 | 2.01 | 2.02 | 2.06 | 2.29 |
| BF16+FA2+TeaCache (0.6)+TorchAO INT8 | 2.60 | 2.41 | 2.66 | 2.68 | 2.68 |

## Key Findings

1. **GPU Performance Comparison**:
   - H100 consistently outperforms A100 across all optimization techniques
   - Average speedup of H100 over A100: ~2.2x

2. **Data Type Impact**:
   - BF16 shows marginal improvement over FP16 on both GPUs
   - A100: ~3% improvement
   - H100: ~1.7% improvement

3. **Attention Optimizations**:
   - FlashAttention2 provides consistent ~11% speedup on A100
   - FlashAttention2 shows minimal impact on H100 (~0.7% improvement)

4. **Caching Strategies**:
   - FirstBlockCache (FBCache) shows the best performance on both GPUs
   - Optimal threshold varies by GPU:
     - A100: 0.35-0.4 threshold (3.3-3.6s)
     - H100: 0.35 threshold (1.87s)
   - TeaCache shows good performance but generally slower than FBCache

5. **Quantization**:
   - TorchAO FP8 shows no significant improvement on H100 in terms of speedup but get 30% reduction in memory

## Additional Key Findings

7. **Advanced Configuration Analysis**:
   - BF16 with FlashAttention2 and FBCache (0.3) provides the most stable performance
   - TorchAO quantization (both FP8 and INT8) shows minimal impact on latency
   - TeaCache configurations consistently show higher latency than FBCache
   - The combination of BF16+FA2+FBCache (0.3) achieves the best balance of speed and stability

8. **Quantization Impact**:
   - FP8 quantization provides marginal improvements (~1-2%) when combined with other optimizations
   - INT8 quantization shows similar performance to FP8
   - Quantization benefits are more pronounced with TeaCache than FBCache

## Updated Recommendations

1. **For A100**:
   - Use FBCache with threshold 0.3 for optimal performance
   - FlashAttention2 provides good standalone improvement

2. **For H100**:
   - FBCache with threshold 0.35 provides best performance
   - FlashAttention2 and quantization show minimal benefits
   - Focus on caching strategies rather than attention optimizations

3. **Best Configs**:
   - Prioritize FBCache with 0.3 threshold as the primary optimization
   - Use BF16 data type with FlashAttention2 for additional ~2-3% improvement
   - Consider TorchAO quantization only if memory constraints are a concern

## Test Configuration

- Model: black-forest-labs/FLUX.1-dev
- Image Resolution: 1024x1024
- Inference Steps: 28
- Guidance Scale: 3.5
- Batch Size: 1 
- Concurrency: 1

## Implementation Notes

- All benchmarks were run with consistent hardware and software configurations
- Each configuration was tested with multiple iterations to ensure reliable results
- Latency measurements include end-to-end pipeline execution

## Future Work

1. **Optimization Opportunities**:
   - Explore Other Quantization Backends.
   - Integrating Sage Attention in Diffusers
   - Test multi-GPU scaling with various configurations
