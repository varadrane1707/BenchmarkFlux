


INFERENCE_PARAMETERS = {
"SQUARE": {
    "prompt": "a tiny astronaut hatching from an egg on the moon",
    "height": 1024,
    "width": 1024,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
},

"LANDSCAPE": {
    "prompt": "A cat sitting on a couch",
    "height": 768,
    "width": 1152,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
},

"PORTRAIT": {
    "prompt": "A potrait of a man with a beard wearing formal attire.",
    "height": 1024,
    "width": 576,
    "num_inference_steps": 28,
    "guidance_scale": 3.5,
}
}


QUANTIZATION_PARAMETERS = {
    "BitsAndBytes": {
        "INT8": {
            "quant_type": "int8",
        },
    },
    "TorchAO": {
        "INT8": {
            "quant_type": "int8dq",
        },
        "FP8": {
            "quant_type": "float8wo_e5m2"
        },
    }
}

ALLOWED_DTYPE = ["FP16", "BF16","FP32"]

ALLOWED_QUANTIZATION_BACKENDS = ["None", "BitsAndBytes", "TorchAO"]

ALLOWED_QUANTIZATION_TYPES = ["None", "INT8", "FP8"]

ALLOWED_ATTENTION_MECHANISMS = ["None", "FlashAttention2"]
ALLOWED_ATTENTION_CACHING = ["None", "TeaCache", "FirstBlockCache"]

ALLOWED_VAE_OPTIMIZATIONS = ["None", "SlicedVAE", "TiledVAE"]

ALLOWED_PARALLEL_GPU_OPTIMIZATIONS = ["None", "ContextParallelismPipeline", "FSDP-XDiT" , "ParallelVAE"]

ALLOWED_COMPILATIONS = ["torch.compile", "None"]

