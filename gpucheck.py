import torch

# 1. Check if a CUDA-compatible GPU is available
if torch.cuda.is_available():
    print("✅ Success! PyTorch can use your GPU.")
    
    # 2. Get the number of available GPUs
    gpu_count = torch.cuda.device_count()
    print(f"   - GPUs Available: {gpu_count}")
    
    # 3. Get the name of the current GPU
    gpu_name = torch.cuda.get_device_name(0) # 0 is the index of the first GPU
    print(f"   - GPU Name: {gpu_name}")

else:
    print("❌ Failure. PyTorch cannot find a CUDA-enabled GPU.")
    print("   - The code will run on the CPU instead.")