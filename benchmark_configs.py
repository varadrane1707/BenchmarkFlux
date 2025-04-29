import json
import time
import random
import numpy as np
import os
from pathlib import Path
from compiledflux import FluxOptimisationConfig, FluxT2I
from utils.parameters import INFERENCE_PARAMETERS

# Create output directories
RESULTS_DIR = Path("benchmark_results")
IMAGES_DIR = RESULTS_DIR / "images"
RESULTS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# Results JSON file
RESULTS_FILE = RESULTS_DIR / "benchmark_stats.json"

# Load existing results if any
existing_results = {}
if RESULTS_FILE.exists():
    with open(RESULTS_FILE, "r") as f:
        existing_results = json.load(f)

# Load all configurations from JSON
with open("configs/complex_configs.json", "r") as f:
    configs = json.load(f)
    
    # #only keep config with name "Dtype_FP8"
    # configs = configs["Dtype_FP8"]
    # configs = {"Dtype_FP8": configs}

# Function to run single inference and measure time
def run_single_inference(flux_t2i, inference_params, save_path=None):
    start_time = time.time()
    output = flux_t2i.inference(inference_params)
    end_time = time.time()
    
    # Save the output image if path is provided
    if save_path:
        output[0].save(save_path)
    
    return end_time - start_time

# Function to calculate statistics
def calculate_statistics(times):
    if not times:
        return None
    
    times = np.array(times)
    stats = {
        'p25': float(np.percentile(times, 25)),  # Convert to float for JSON serialization
        'p50': float(np.percentile(times, 50)),
        'p90': float(np.percentile(times, 90)),
        'p99': float(np.percentile(times, 99)),
        'avg': float(np.mean(times)),
        'min': float(np.min(times)),
        'total_iterations': len(times)
    }
    return stats

# Time limit per configuration (3 minutes)
TIME_LIMIT_SECONDS = 90  # 90 seconds

# Results storage
results = existing_results.copy()

# Run benchmarks for each config
for config_name, config_params in configs.items():
    # Skip if already benchmarked
    if config_name in results:
        print(f"\nSkipping already benchmarked configuration: {config_name}")
        continue
        
    print(f"\nTesting configuration: {config_name}")
    print("-" * 50)
    
    # Create config-specific image directory
    config_image_dir = IMAGES_DIR / config_name
    config_image_dir.mkdir(exist_ok=True)
    
    try:
        # Initialize FluxOptimisationConfig
        flux_config = FluxOptimisationConfig(**config_params)
        
        #clear GPU memory and processes
        import torch
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        
        # Initialize FluxT2I
        flux_t2i = FluxT2I(flux_config)
        flux_t2i.compile_pipeline()
        
        # Storage for timing results
        inference_times = []
        
        # Track configuration start time
        config_start_time = time.time()
        iteration = 0
        
        # Run iterations until time limit
        while (time.time() - config_start_time) < TIME_LIMIT_SECONDS:
            iteration += 1
            
            # Randomly select inference parameters
            inference_type = random.choice(list(INFERENCE_PARAMETERS.keys()))
            inference_params = INFERENCE_PARAMETERS[inference_type].copy()
            
            # Force num_images_per_prompt to 1
            inference_params['num_images_per_prompt'] = 1
            
            try:
                # Set up image save path
                image_path = config_image_dir / f"iter_{iteration}_{inference_type}.png"
                
                # Run inference and measure time
                inference_time = run_single_inference(flux_t2i, inference_params, image_path)
                inference_times.append(inference_time)
                
                elapsed = time.time() - config_start_time
                print(f"Iteration {iteration}: {inference_time:.4f}s (Total elapsed: {elapsed:.1f}s)")
                
            except Exception as e:
                print(f"Error in iteration {iteration}: {str(e)}")
                continue
        
        # Calculate statistics
        stats = calculate_statistics(inference_times)
        if stats:
            # Store configuration parameters along with stats
            results[config_name] = {
                'stats': stats,
                'config_params': config_params,
                'last_inference_params': inference_params
            }
            
            # Save results after each configuration
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=4)
            
            print(f"\nResults for {config_name}:")
            print(f"Total iterations: {stats['total_iterations']}")
            print(f"p25: {stats['p25']:.4f}s")
            print(f"p50: {stats['p50']:.4f}s")
            print(f"p90: {stats['p90']:.4f}s")
            print(f"p99: {stats['p99']:.4f}s")
            print(f"avg: {stats['avg']:.4f}s")
            print(f"min: {stats['min']:.4f}s")
            
    except Exception as e:
        print(f"Failed to test configuration {config_name}: {str(e)}")
        continue

# Find the fastest configuration
if results:
    fastest_config = min(results.items(), key=lambda x: x[1]['stats']['min'])
    print("\n" + "="*50)
    print(f"Fastest configuration: {fastest_config[0]}")
    print(f"Best inference time: {fastest_config[1]['stats']['min']:.4f}s")
    print("="*50)
    
    