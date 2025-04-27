from itertools import product
from typing import Dict, List
from parameters import (
    INFERENCE_PARAMETERS,
    ALLOWED_DTYPE,
    ALLOWED_QUANTIZATION_BACKENDS,
    ALLOWED_QUANTIZATION_TYPES,
    ALLOWED_ATTENTION_MECHANISMS,
    ALLOWED_ATTENTION_CACHING,
    ALLOWED_VAE_OPTIMIZATIONS,
    ALLOWED_PARALLEL_GPU_OPTIMIZATIONS,
    ALLOWED_COMPILATIONS,
    QUANTIZATION_PARAMETERS
)

def generate_benchmark_configs() -> List[Dict]:
    """
    Generate all possible combinations of inference configurations for benchmarking.
    """
    base_configs = []
    
    # Generate combinations for each image type (SQUARE, LANDSCAPE, PORTRAIT)
    for image_type, base_params in INFERENCE_PARAMETERS.items():
        # Create combinations of all possible parameters
        combinations = product(
            ALLOWED_DTYPE,
            ALLOWED_QUANTIZATION_BACKENDS,
            ALLOWED_QUANTIZATION_TYPES,
            ALLOWED_ATTENTION_MECHANISMS,
            ALLOWED_ATTENTION_CACHING,
            ALLOWED_VAE_OPTIMIZATIONS,
            ALLOWED_PARALLEL_GPU_OPTIMIZATIONS,
            ALLOWED_COMPILATIONS
        )
        
        for combo in combinations:
            (dtype, quant_backend, quant_type, attention_mech, 
             attention_cache, vae_opt, parallel_opt, compilation) = combo
            
            # Skip invalid quantization combinations
            if quant_type not in QUANTIZATION_PARAMETERS.get(quant_backend, {}):
                continue
                
            config = {
                "image_type": image_type,
                "base_parameters": base_params.copy(),
                "optimizations": {
                    "dtype": dtype,
                    "quantization": {
                        "backend": quant_backend,
                        "type": quant_type,
                        **QUANTIZATION_PARAMETERS[quant_backend][quant_type]
                    },
                    "attention": {
                        "mechanism": attention_mech,
                        "caching": attention_cache
                    },
                    "vae_optimization": vae_opt,
                    "parallel_optimization": parallel_opt,
                    "compilation": compilation
                }
            }
            base_configs.append(config)
    
    return base_configs

def get_config_name(config: Dict) -> str:
    """Generate a unique name for a configuration."""
    opt = config["optimizations"]
    return f"{config['image_type']}__{opt['dtype']}__{opt['quantization']['backend']}__{opt['quantization']['type']}__{opt['attention']['mechanism']}__{opt['attention']['caching']}__{opt['vae_optimization']}__{opt['parallel_optimization']}__{opt['compilation']}".replace(".", "_") 