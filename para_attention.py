"""
Implements parallel attention mechanisms for Flux.
Original Implementation: https://github.com/chengzeyi/ParaAttention/tree/main

#installation 
pip3 install para-attn
"""
from diffusers import FluxPipeline
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

def FBcache(pipeline : FluxPipeline):
    return apply_cache_on_pipe(pipeline , threshold=1)
    
        
        
def ContextParallelismPipeline(pipeline : FluxPipeline):
    mesh = init_context_parallel_mesh(
        pipeline.device.type,
        max_ring_dim_size=2,
    )
    return parallelize_pipe(pipeline, mesh=mesh)


def ParallelVAE(pipeline : FluxPipeline):
    mesh = init_context_parallel_mesh(
        pipeline.device.type,
        max_ring_dim_size=2,
    )
    return parallelize_vae(pipeline.vae, mesh=mesh._flatten())
    