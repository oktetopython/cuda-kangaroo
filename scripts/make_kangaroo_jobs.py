#!/usr/bin/env python3
"""
GPU任务配置生成器
将GLV预过滤结果转换为Kangaroo GPU任务
"""

import os
import sys
import json
import argparse
from datetime import datetime

def detect_gpu_config():
    """检测GPU配置"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    name, memory = line.split(', ')
                    gpus.append({
                        "name": name.strip(),
                        "memory_mb": int(memory.strip())
                    })
            return gpus
    except:
        pass
    
    # 默认配置（基于用户的NVIDIA H20）
    return [
        {"name": "NVIDIA H20", "memory_mb": 96000}  # 假设96GB显存
    ]

def calculate_optimal_params(gpu_memory_mb, subinterval_size_bits):
    """根据GPU显存计算最优参数"""
    # 基础参数
    base_threads = 1024 * 64  # 65536
    base_batch_size = 2**20   # 1M
    
    # 根据显存调整
    if gpu_memory_mb >= 80000:  # 80GB+
        gpu_threads = base_threads * 2
        batch_size = base_batch_size * 4
        dp_bits = 12
    elif gpu_memory_mb >= 40000:  # 40GB+
        gpu_threads = base_threads
        batch_size = base_batch_size * 2
        dp_bits = 11
    else:  # <40GB
        gpu_threads = base_threads // 2
        batch_size = base_batch_size
        dp_bits = 10
    
    # 根据子区间大小调整DP
    if subinterval_size_bits <= 50:
        dp_bits -= 2
    elif subinterval_size_bits >= 60:
        dp_bits += 1
    
    return {
        "gpu_threads": gpu_threads,
        "batch_size": batch_size,
        "dp_bits": dp_bits
    }

def generate_kangaroo_jobs(subintervals_file, output_file="jobs.json"):
    """生成Kangaroo GPU任务配置"""
    print("🔧 生成Kangaroo GPU任务配置")
    print("=" * 50)
    
    # 读取子区间数据
    with open(subintervals_file, 'r') as f:
        data = json.load(f)
    
    subintervals = data["subintervals"]
    pubkey = data["pubkey"]
    
    print(f"📊 输入数据:")
    print(f"   子区间文件: {subintervals_file}")
    print(f"   子区间数量: {len(subintervals)}")
    print(f"   目标公钥: {pubkey}")
    
    # 检测GPU配置
    gpus = detect_gpu_config()
    print(f"   检测到GPU: {len(gpus)}张")
    for i, gpu in enumerate(gpus):
        print(f"     GPU{i}: {gpu['name']} ({gpu['memory_mb']/1024:.1f}GB)")
    
    # 生成任务配置
    jobs = []
    gpu_count = len(gpus)
    
    for i, interval in enumerate(subintervals):
        gpu_id = i % gpu_count  # 轮询分配GPU
        gpu_info = gpus[gpu_id]
        
        # 计算最优参数
        params = calculate_optimal_params(gpu_info["memory_mb"], interval["size_bits"])

        # 计算GPU网格大小 (基于RTX 2080 Ti: 68 SM)
        grid_x = min(64, params["gpu_threads"] // 1024)
        grid_y = min(64, params["gpu_threads"] // grid_x // 16)
        grid_size = f"{grid_x},{grid_y}"

        job = {
            "job_id": f"puzzle135_interval_{interval['id']}",
            "gpu_id": gpu_id,
            "pubkey": pubkey,
            "range_start": interval["start"],
            "range_end": interval["end"],
            "size_bits": interval["size_bits"],
            "estimated_time_hours": interval["estimated_time_hours"],
            "kangaroo_params": {
                "threads": params["gpu_threads"],
                "dp_bits": params["dp_bits"],
                "batch_size": params["batch_size"],
                "grid_size": grid_size,
                "work_file": f"work/puzzle135_interval_{interval['id']}.work",
                "log_file": f"logs/puzzle135_interval_{interval['id']}.log",
                "result_file": f"results/puzzle135_interval_{interval['id']}.result"
            },
            "config_file": f"configs/puzzle135_interval_{interval['id']}.txt",
            "command_template": "build/Release/kangaroo.exe -t 0 -gpu -d {dp_bits} -g {grid_size} -w {work_file} -o {result_file} {config_file}",
            "priority": 1.0 - (i / len(subintervals)),  # 前面的区间优先级更高
            "status": "pending",
            "created_at": datetime.now().isoformat()
        }
        
        jobs.append(job)
    
    # 创建必要的目录
    os.makedirs("work", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    # 生成配置文件
    print(f"\n📝 生成配置文件")
    for job in jobs:
        config_file = job["config_file"]
        with open(config_file, 'w') as f:
            f.write(f"{job['range_start']}\n")
            f.write(f"{job['range_end']}\n")
            f.write(f"{job['pubkey']}\n")
        print(f"   {config_file}")

    
    # 生成任务配置文件
    job_config = {
        "puzzle": 135,
        "total_jobs": len(jobs),
        "gpu_count": gpu_count,
        "generated_at": datetime.now().isoformat(),
        "source_file": subintervals_file,
        "jobs": jobs
    }
    
    with open(output_file, 'w') as f:
        json.dump(job_config, f, indent=2)
    
    print(f"\n✅ 任务配置生成完成！")
    print(f"   输出文件: {output_file}")
    print(f"   总任务数: {len(jobs)}")
    print(f"   GPU分配:")
    
    for gpu_id in range(gpu_count):
        gpu_jobs = [j for j in jobs if j["gpu_id"] == gpu_id]
        print(f"     GPU{gpu_id}: {len(gpu_jobs)} 个任务")
    
    # 生成启动脚本
    generate_launch_scripts(job_config)
    
    return output_file

def generate_launch_scripts(job_config):
    """生成启动脚本"""
    print(f"\n🚀 生成启动脚本")
    
    # Windows批处理脚本
    with open("start_puzzle135.bat", 'w') as f:
        f.write("@echo off\n")
        f.write("echo Starting Bitcoin Puzzle 135 GPU Solving...\n")
        f.write("echo.\n")
        
        for job in job_config["jobs"]:
            params = job["kangaroo_params"]
            cmd = job["command_template"].format(
                dp_bits=params["dp_bits"],
                grid_size=params["grid_size"],
                work_file=params["work_file"],
                result_file=params["result_file"],
                config_file=job["config_file"]
            )
            f.write(f'start "GPU{job["gpu_id"]}_Job{job["job_id"]}" {cmd}\n')
        
        f.write("echo All jobs started!\n")
        f.write("pause\n")
    
    # Linux shell脚本
    with open("start_puzzle135.sh", 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("echo 'Starting Bitcoin Puzzle 135 GPU Solving...'\n")
        f.write("echo\n")
        
        for job in job_config["jobs"]:
            params = job["kangaroo_params"]
            cmd = job["command_template"].format(
                dp_bits=params["dp_bits"],
                grid_size=params["grid_size"],
                work_file=params["work_file"],
                result_file=params["result_file"],
                config_file=job["config_file"]
            )
            f.write(f'{cmd} > {params["log_file"]} 2>&1 &\n')
        
        f.write("echo 'All jobs started!'\n")
        f.write("wait\n")
    
    os.chmod("start_puzzle135.sh", 0o755)
    
    print(f"   Windows脚本: start_puzzle135.bat")
    print(f"   Linux脚本: start_puzzle135.sh")

def main():
    parser = argparse.ArgumentParser(description="生成Kangaroo GPU任务配置")
    parser.add_argument("subintervals_file", help="GLV预过滤结果文件")
    parser.add_argument("-o", "--output", default="jobs.json", help="输出任务配置文件")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.subintervals_file):
        print(f"❌ 子区间文件不存在: {args.subintervals_file}")
        return 1
    
    try:
        output_file = generate_kangaroo_jobs(args.subintervals_file, args.output)
        print(f"\n🎉 任务配置生成成功！")
        print(f"\n🚀 下一步:")
        print(f"   Windows: start_puzzle135.bat")
        print(f"   Linux: ./start_puzzle135.sh")
        print(f"   手动: 查看 {args.output} 配置文件")
        return 0
    except Exception as e:
        print(f"❌ 生成任务配置失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
