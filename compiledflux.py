import time
import logging
import torch
from typing import Optional , Literal
from pydantic import BaseModel

from parameters import INFERENCE_PARAMETERS, QUANTIZATION_PARAMETERS
from parameters import (ALLOWED_ATTENTION_MECHANISMS, 
                        ALLOWED_ATTENTION_CACHING, 
                        ALLOWED_QUANTIZATION_BACKENDS, 
                        ALLOWED_QUANTIZATION_TYPES, 
                        ALLOWED_VAE_OPTIMIZATIONS, 
                        ALLOWED_PARALLEL_GPU_OPTIMIZATIONS,
                        ALLOWED_COMPILATIONS,
                        ALLOWED_DTYPE,
                        ALLOWED_COMPILATIONS
                        )

from diffusers import FluxPipeline, FluxControlNetPipeline, FluxControlNetModel
from diffusers import AutoencoderKL,FluxTransformer2DModel,FlowMatchEulerDiscreteScheduler
from transformers import CLIPTextModel,T5EncoderModel,CLIPTokenizer,T5TokenizerFast
        
        
class FluxOptimisationConfig(BaseModel):
    """
    Example JSONConfig:
    {
        "model_id": "flux-1-dev",
        "device": "cuda",
        "is_controlnet": False,
        "controlnet_model_id": None,
        "quantize": True,
        "dtype": "FP16",
        "quantize_dtype": "INT8",
        "quantize_backend": "BitsAndBytes",
        "attention_mechanism": "SageAttention2",
        "attention_caching": "TeaCache",
        "vae_optimizations": "TiledVAE",
        "parallel_gpu_setup": False,
        "parallel_gpu_optimizations": None,
        "compilations": None
    }
    """
    
    model_id: str
    device: str
    
    is_controlnet: bool
    controlnet_model_id: Optional[str]
    
    quantize: bool
    dtype: Literal[tuple(ALLOWED_DTYPE)]
    quantize_dtype: Literal[tuple(ALLOWED_QUANTIZATION_TYPES)]
    quantize_backend: Literal[tuple(ALLOWED_QUANTIZATION_BACKENDS)]
    
    attention_mechanism: Literal[tuple(ALLOWED_ATTENTION_MECHANISMS)]
    attention_caching: Literal[tuple(ALLOWED_ATTENTION_CACHING)]
    
    vae_optimizations: Optional[Literal[tuple(ALLOWED_VAE_OPTIMIZATIONS)]]
    
    parallel_gpu_setup: bool
    parallel_gpu_optimizations: Optional[Literal[tuple(ALLOWED_PARALLEL_GPU_OPTIMIZATIONS)]]
    compilations: Optional[Literal[tuple(ALLOWED_COMPILATIONS)]]
    

class FluxT2I():
    def __init__(self, config: FluxOptimisationConfig):
        self.device = config.device
        self.is_controlnet = config.is_controlnet
        self.model_id = config.model_id
        self.controlnet_model_id = config.controlnet_model_id
        self.dtype = config.dtype
        
        self.is_quantize = config.quantize
        self.quantize_dtype = config.quantize_dtype
        self.quantize_backend = config.quantize_backend
        
        self.attention_mechanism = config.attention_mechanism
        self.attention_caching = config.attention_caching
        
        self.vae_optimizations = config.vae_optimizations
        
        self.parallel_gpu_setup = config.parallel_gpu_setup
        self.parallel_gpu_optimizations = config.parallel_gpu_optimizations
        
        self.compilations = config.compilations
        

    def soft_validations(self):
        if self.is_quantize:
            if self.quantize_dtype not in ALLOWED_QUANTIZATION_TYPES:
                raise ValueError("quantize_dtype must be INT8 or INT4 or FP8")
            
            if self.quantize_backend not in ALLOWED_QUANTIZATION_BACKENDS:
                raise ValueError("quantize_backend must be BitsAndBytes or TorchAO or GGUF or Quanto")
            
            if self.dtype not in ALLOWED_DTYPE:
                raise ValueError("dtype must be FP16 or BF16 or TF32 or FP8 or FP32")
            
            if self.inference_optimizations not in ALLOWED_INFERENCE_OPTIMIZATIONS:
                raise ValueError("inference_optimizations must be SageAttention2 or FlashAttention or TeaCache or FirstBlockCache")
            
            if self.vae_optimizations not in ALLOWED_VAE_OPTIMIZATIONS:
                raise ValueError("vae_optimizations must be SlicedVAE or TiledVAE or ParallelVAE")
            
            if self.parallel_gpu_optimizations not in ALLOWED_PARALLEL_GPU_OPTIMIZATIONS:
                raise ValueError("parallel_gpu_optimizations must be ContextParallelismPipeline or FSDP-XDiT")
            
            if self.compilations not in ALLOWED_COMPILATIONS:
                raise ValueError("compilations must be torch.compile or tensorrt compilation")
            
            if not self.parallel_gpu_setup:
                if self.parallel_gpu_optimizations is not None:
                    raise ValueError("Parallel GPU Optimizations are not supported without parallel_gpu_setup")
                
                if self.vae_optimizations == "ParallelVAE":
                    raise ValueError("ParallelVAE is not supported without parallel_gpu_setup")
                
            return
        
        
    def cross_validation(self):
        pass
    
    def generate_quantization_config(self):
        self.quantization_config = None
        if self.quantize_backend == "BitsAndBytes":
            try:
                import bitsandbytes
            except ImportError:
                raise ImportError("bitsandbytes is not installed. Please install it with `pip install bitsandbytes`")
            if self.quantize_dtype == "INT8":
                self.quantization_config = bitsandbytes.optim.QuantizationConfig(load_in_8bit=True) 
            else :
                raise ValueError("INT8 is the only supported quantization type for BitsAndBytes")
        elif self.quantize_backend == "TorchAO":
            try:
                from torchao.quantization import autoquant
            except ImportError:
                raise ImportError("torchao is not installed. Please install it with `pip install torchao`")
            if self.quantize_dtype == "INT8":
                self.quantization_config = autoquant.AutoQuantConfig(quant_type=autoquant.QuantType.INT8)
            elif self.quantize_dtype == "FP8":
                self.quantization_config = autoquant.AutoQuantConfig(quant_type=autoquant.QuantType.FP8)
            else :
                raise ValueError("INT8 or FP8 is the only supported quantization type for TorchAO")  
        return 
    
    def load_pipeline(self):
        if self.is_controlnet:
            controlnet = FluxControlNetModel.from_pretrained(
                self.controlnet_model_id, torch_dtype=self.dtype
            ) 
            
            self.pipeline = FluxControlNetPipeline.from_pretrained(
                self.model_id,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                quantization_config=self.quantization_config
            ).to(self.device)
        else:
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                quantization_config=self.quantization_config
            ).to(self.device)
        return 
    
    def enable_attention_caching(self):
        if self.attention_caching == "TeaCache":
            from teacache import teacache_forward
            self.pipeline.transformer.forward = teacache_forward
            self.pipeline.transformer.__class__.enable_teacache = True
            self.pipeline.transformer.__class__.cnt = 0
            self.pipeline.transformer.__class__.num_steps = num_inference_steps
            self.pipeline.transformer.__class__.rel_l1_thresh = 0.6 # 0.25 for 1.5x speedup, 0.4 for 1.8x speedup, 0.6 for 2.0x speedup, 0.8 for 2.25x speedup
            self.pipeline.transformer.__class__.accumulated_rel_l1_distance = 0
            self.pipeline.transformer.__class__.previous_modulated_input = None
            self.pipeline.transformer.__class__.previous_residual = None
            return
        elif self.attention_caching == "FirstBlockCache":
            from para_attention import FBcache
            self.pipeline = FBcache(self.pipeline)
            
            
    def compile_pipeline(self):
        self.soft_validations()
        if self.is_quantize:
            self.generate_quantization_config()
        else:
            self.quantization_config = None
            
        self.load_pipeline()
        
        if self.attention_mechanism == "SageAttention2":
            pass
        elif self.attention_mechanism == "FlashAttention":
            pass
            
        if not self.attention_caching == "None":
            self.enable_attention_caching()
            
        if self.vae_optimizations == "TiledVAE":
            self.pipeline.enable_tiled_vae()
            
        if self.compilations == "torch.compile":
            self.pipeline = torch.compile(self.pipeline, mode="max-autotune")
              
        # if self.parallel_gpu_setup:
        #     if self.parallel_gpu_optimizations == "ContextParallelismPipeline":
        #         self.pipeline.enable_context_parallelism_pipeline()
        #     elif self.parallel_gpu_optimizations == "FSDP-XDiT":
        #         self.pipeline.enable_fsdp_xdit()
            
            
        #     if self.parallel_gpu_optimizations == "ParallelVAE":
        #         self.pipeline.enable_parallel_vae()
         
        return 
        

                
    def inference(self ,inference_config: dict):
        output = self.pipeline(
            prompt=inference_config["prompt"],
            num_inference_steps=inference_config["num_inference_steps"],
            guidance_scale=inference_config["guidance_scale"],
            height=inference_config["height"],
            width=inference_config["width"],
            num_images_per_prompt=inference_config["num_images_per_prompt"],  
        ).images
        
        return output
        
        
        
            
        
            
    
            

        