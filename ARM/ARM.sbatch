#!/bin/sh

#SBATCH --job-name=arm-tmp-trend
#SBATCH --partition=general
#SBATCH --qos=medium 
#SBATCH --time=26:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

module use /opt/insy/modulefiles
module load cuda/11.5 cudnn/11.5-8.3.0.98
pip3 install --user --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 >/dev/null
pip3 install --user matplotlib >/dev/null
pip3 install --user seaborn >/dev/null
pip3 install --user nltk >/dev/null
pip3 install --user spacey >/dev/null
pip3 install --user flair >/dev/null

echo "Starting training!"
python3 ARM.py
echo "Finishing training!"

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
