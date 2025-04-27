from compiledflux import FluxOptimisationConfig

flux_config = FluxOptimisationConfig(
    model_id="black-forest-labs/FLUX.1-dev",
    device="cuda",
    is_controlnet=False,
    controlnet_model_id=None,
    quantize=False,
    dtype="FP16",
    quantize_dtype="INT8",
    quantize_backend="BitsAndBytes",
    attention_mechanism="SageAttention2",
    attention_caching="TeaCache",
    vae_optimizations="None",
    parallel_gpu_setup=False,
    parallel_gpu_optimizations=None,
    compilations=None
)

print(flux_config)

from compiledflux import FluxT2I

flux_t2i = FluxT2I(flux_config)
flux_t2i.compile_pipeline()


inference_config = {
    "prompt": "A beautiful landscape with a river and mountains",
    "num_inference_steps": 28,
    "guidance_scale": 7.5,
    "height": 1024,
    "width": 1024,
    "num_images_per_prompt": 4
}
# flux_t2i.inference(
#     prompt="A beautiful landscape with a river and mountains",
# )
# print(flux_t2i.pipeline)

output = flux_t2i.inference(inference_config)
for i, image in enumerate(output):
    image.save(f"output_{i}.png")