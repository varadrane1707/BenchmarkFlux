from compiledflux import FluxOptimisationConfig , FluxT2I

flux_config = FluxOptimisationConfig(
    model_id="black-forest-labs/FLUX.1-dev",
    device="cuda",
    is_controlnet=False,
    controlnet_model_id=None,
    quantize=False,
    dtype="BF16",
    quantize_dtype="None",
    quantize_backend="None",
    attention_mechanism="FlashAttention2",
    attention_caching="TeaCache",
    vae_optimizations="None",
    parallel_gpu_setup=False,
    parallel_gpu_optimizations=None,
    compilations="torch.compile",
    caching_threshold=0.35
)

print(flux_config)

flux_t2i = FluxT2I(flux_config)
flux_t2i.compile_pipeline()


from BenchmarkFlux.utils.parameters import INFERENCE_PARAMETERS
import random
import time
inference_type = random.choice(list(INFERENCE_PARAMETERS.keys()))
inference_params = INFERENCE_PARAMETERS[inference_type].copy()
            
inference_params['num_images_per_prompt'] = 1 

time_start = time.time()
output = flux_t2i.inference(
    inference_params
)
output[0].save("output.png")
print(f"Time taken: {time.time() - time_start:.2f} seconds")
