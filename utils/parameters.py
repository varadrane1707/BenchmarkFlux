


INFERENCE_PARAMETERS = {
"astronaut": {
    "prompt": "a tiny astronaut hatching from an egg on the moon",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
},

"cat": {
    "prompt": "A cat sitting on a couch",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
},

"potrait": {
    "prompt": "A potrait of a man with a beard wearing formal attire.",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
}
}


QUANTIZATION_PARAMETERS = {
    
    "TorchAO": {
        "FP8": {
            "quant_type": "float8wo"
        },
        "INT8": {
            "quant_type": "int8wo"
        },
    }
}

ALLOWED_DTYPE = ["FP16", "BF16","FP32","FP8"]

ALLOWED_QUANTIZATION_BACKENDS = ["None", "TorchAO"]

ALLOWED_QUANTIZATION_TYPES = ["None", "FP8", "INT8"]

ALLOWED_ATTENTION_MECHANISMS = ["None", "FlashAttention2"]
ALLOWED_ATTENTION_CACHING = ["None", "TeaCache", "FirstBlockCache"]

ALLOWED_VAE_OPTIMIZATIONS = ["None", "TiledVAE"]

ALLOWED_PARALLEL_GPU_OPTIMIZATIONS = ["None", "ContextParallelismPipeline", "FSDP-XDiT" , "ParallelVAE"]

ALLOWED_COMPILATIONS = ["torch.compile", "None"]

