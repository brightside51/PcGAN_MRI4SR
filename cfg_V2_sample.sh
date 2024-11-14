#!/bin/bash
#
#SBATCH --partition=gpu_min24gb                                   # Partition (check with "$sinfo")
#SBATCH --output=../../MetaBreast/logs/pcgan_sr/V2/output_sample.out           # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --error=../../MetaBreast/logs/pcgan_sr/V2/error_sample.err             # (Optional) Filename with STDERR. If ommited, use STDOUT.
#SBATCH --job-name=pcgan_sr                                        # (Optional) Job name
#SBATCH --time=14-00:00                                             # (Optional) Time limit (D: days, HH: hours, MM: minutes)
#SBATCH --qos=gpu_min24GB                                           # (Optional) 01.ctm-deep-05

python infer_V2.py