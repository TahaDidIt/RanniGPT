import torch
import os
import ctypes

print("CUDA Available:", torch.cuda.is_available())
print("Torch CUDA Version:", torch.version.cuda)
print("Torch Compiled with CUDA:", torch.__version__)

# Check where CUDA is loaded from
cuda_path = torch.utils.cpp_extension.CUDA_HOME
print("CUDA HOME:", cuda_path)

# Check if the nvrtc.dll file exists
nvrtc_dll_path = os.path.join(cuda_path, "bin", "nvrtc-builtins64_122.dll") if cuda_path else "Not Found"
print("Expected NVRTC Path:", nvrtc_dll_path)

# Try manually loading the DLL
try:
    ctypes.WinDLL(nvrtc_dll_path)
    print("nvrtc DLL Loaded Successfully")
except OSError as e:
    print("Failed to load nvrtc:", e)
