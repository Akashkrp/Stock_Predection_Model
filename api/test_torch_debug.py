
import os
import sys
import ctypes

# Path to torch lib
torch_lib = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "torch", "lib")
print(f"Torch Lib Path: {torch_lib}")

dll_to_load = os.path.join(torch_lib, "libiomp5md.dll")
try:
    if os.path.exists(dll_to_load):
        print(f"Loading {dll_to_load}...")
        ctypes.CDLL(dll_to_load)
        print("Successfully loaded libiomp5md.dll")
    else:
        print(f"{dll_to_load} does not exist")
except Exception as e:
    print(f"Failed to load libiomp5md.dll: {e}")

try:
    import torch
    print("Torch imported successfully")
except Exception as e:
    print(f"Error importing torch: {e}")
