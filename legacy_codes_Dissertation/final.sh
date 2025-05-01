#!/bin/bash
#SBATCH -c 4                             # Request 160 logical CPU cores
#SBATCH --mem=16g                        # Request 64GB of system memory
#SBATCH --gres=gpu:1                     # Request 8 GPUs
#SBATCH -p cspg                          # Assign the job to the cspg partition
#SBATCH --qos=cspg                       # Indicate to run the job under the cspg QoS
#SBATCH --output=outputs/slurm-%j.out    # Capture standard output to a file named slurm-<jobid>.out
#SBATCH --error=outputs/slurm-%j.err     # Capture standard error to a file named slurm-<jobid>.err


module load nvidia/cuda-11.7
module load nvidia/cudnn-v8.1.1.33-forcuda11.0-to-11.2

source ~/miniconda3/etc/profile.d/conda.sh


conda activate torch3


python plantvill_torchbinary_crossvall.py --epochs 50 --model resnet18
python plantvill_torchbinary_crossvall.py --epochs 50 --model resnet50
python plantvill_torchbinary_crossvall.py --epochs 50 --model resnet152

python plantvill_torchbinary.py --epochs 50 --model resnet18
python plantvill_torchbinary.py --epochs 50 --model resnet50
python plantvill_torchbinary.py --epochs 50 --model resnet152
