#!/bin/bash
echo "After rebooting, run this script to verify CUDA is working:"
echo ""
echo "1. Check nvidia-smi:"
nvidia-smi
echo ""
echo "2. Test PyTorch CUDA:"
cd /home/gowtham/7th\ sem/MP/Project
source venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
