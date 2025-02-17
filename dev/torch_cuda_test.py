'''
import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
'''
import os
print("CUDA Path:", os.getenv("CUDA_PATH"))
print("CUDA Bin Directory:", os.getenv("CUDA_BIN_PATH"))
print("Library Path:", os.getenv("PATH"))

import sys
print(sys.executable)
import torch
print(torch.__file__) 
print(torch.cuda.is_available())
from torch.utils import collect_env
print(collect_env.main())