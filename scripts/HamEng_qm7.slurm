#!/bin/bash
#SBATCH -J HE-QM7             # 作业名是 test
#SBATCH -p GPU               # 提交到 cpu 分区
#SBATCH -N 1                 # 使用一个节点
#SBATCH --cpus-per-task=4    # 每个进程占用一个 cpu 核心
#SBATCH -t 120:00:00          # 任务最大运行时间是 5 分钟
#SBATCH -o outputs/hameng_qm7-%j.txt        # 将屏幕的输出结果保存到当前文件夹的 test.out
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --ntasks-per-node=1

# module add anaconda/3-5.0.0.1
# source activate ysw_pytorch
python HamEng_qm7.py --seed 16880611
python HamEng_qm7.py --seed 17760704
python HamEng_qm7.py --seed 17890714
python HamEng_qm7.py --seed 19491001
python HamEng_qm7.py --seed 19900612
