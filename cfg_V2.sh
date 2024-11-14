#!/bin/bash
#!/bin/bash
#
#SBATCH --partition=gpu_min80gb                                   # Partition (check with "$sinfo")
#SBATCH --output=../../MetaBreast/logs/pcgan_sr/V2/output.out           # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --error=../../MetaBreast/logs/pcgan_sr/V2/error.err             # (Optional) Filename with STDERR. If ommited, use STDOUT.
#SBATCH --job-name=pcgan_sr                                        # (Optional) Job name
#SBATCH --time=14-00:00                                             # (Optional) Time limit (D: days, HH: hours, MM: minutes)
#SBATCH --qos=gpu_min80GB                                           # (Optional) 01.ctm-deep-05

python train_V2.py