import sys
import torch
import transformers
import peft
import bitsandbytes
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('Transformers:', transformers.__version__)
print('PEFT:', peft.__version__)
print('bitsandbytes:', bitsandbytes.__version__)
print('CUDA tersedia:', torch.cuda.is_available())
print('Runtime CUDA PyTorch:', torch.version.cuda)
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
else:
    raise SystemExit('Hentikan di sini. GPU belum terdeteksi.')
