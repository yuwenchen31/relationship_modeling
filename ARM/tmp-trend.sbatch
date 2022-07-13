#!/bin/sh

#SBATCH --job-name=us-ru
#SBATCH --partition=general
#SBATCH --qos=short 
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:pascal:1

module use /opt/insy/modulefiles
module load cuda/11.5 cudnn/11.5-8.3.0.98
pip3 install --user --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113 >/dev/null
pip3 install --user matplotlib >/dev/null
pip3 install --user seaborn >/dev/null
pip3 install --user nltk >/dev/null
pip3 install --user spacey >/dev/null
pip3 install --user flair >/dev/null

echo "Start getting the plot!"
python3 TemporalTrend.py
echo "Finish temporal trend plotting!"

/usr/bin/nvidia-smi --query-accounted-apps='gpu_utilization,mem_utilization,max_memory_usage,time' --format='csv' | /usr/bin/grep -v -F "$previous"
