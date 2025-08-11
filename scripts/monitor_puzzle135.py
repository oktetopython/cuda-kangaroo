#!/usr/bin/env python3
"""
Bitcoin Puzzle 135 实时监控系统
监控GPU任务进度、性能和结果
"""

import os
import sys
import json
import time
import re
import subprocess
from datetime import datetime, timedelta
from collections import defaultdict

class Puzzle135Monitor:
    def __init__(self, jobs_file="jobs.json"):
        self.jobs_file = jobs_file
        self.jobs = {}
        self.load_jobs()
        self.start_time = datetime.now()
        
    def load_jobs(self):
        """加载任务配置"""
        if os.path.exists(self.jobs_file):
            with open(self.jobs_file, 'r') as f:
                data = json.load(f)
                for job in data["jobs"]:
                    self.jobs[job["job_id"]] = job
        
    def get_gpu_status(self):
        """获取GPU状态"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        gpus.append({
                            "id": int(parts[0]),
                            "name": parts[1],
                            "utilization": int(parts[2]),
                            "memory_used": int(parts[3]),
                            "memory_total": int(parts[4]),
                            "temperature": int(parts[5])
                        })
                return gpus
        except:
            pass
        return []
    
    def parse_log_file(self, log_file):
        """解析日志文件获取进度"""
        if not os.path.exists(log_file):
            return {"status": "not_started", "progress": 0, "speed": 0, "errors": []}
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # 解析关键信息
            status = "running"
            progress = 0
            speed = 0
            errors = []
            
            # 查找速度信息 (MK/s)
            speed_matches = re.findall(r'(\d+\.?\d*)\s*MK/s', content)
            if speed_matches:
                speed = float(speed_matches[-1])
            
            # 查找进度信息
            progress_matches = re.findall(r'(\d+\.?\d*)%', content)
            if progress_matches:
                progress = float(progress_matches[-1])
            
            # 查找错误信息
            error_patterns = [
                r'ERROR:.*',
                r'CUDA error.*',
                r'out of memory.*',
                r'Warning.*lost'
            ]
            for pattern in error_patterns:
                error_matches = re.findall(pattern, content, re.IGNORECASE)
                errors.extend(error_matches)
            
            # 检查是否找到结果
            if "Private Key Found" in content or "FOUND" in content:
                status = "found"
                progress = 100
            elif "completed" in content.lower():
                status = "completed"
                progress = 100
            elif errors:
                status = "error"
            
            return {
                "status": status,
                "progress": progress,
                "speed": speed,
                "errors": errors[-5:],  # 只保留最近5个错误
                "last_update": datetime.fromtimestamp(os.path.getmtime(log_file))
            }
            
        except Exception as e:
            return {"status": "error", "progress": 0, "speed": 0, "errors": [str(e)]}
    
    def check_results(self):
        """检查是否有结果文件"""
        found_results = []
        for job_id, job in self.jobs.items():
            result_file = job["kangaroo_params"]["result_file"]
            if os.path.exists(result_file):
                try:
                    with open(result_file, 'r') as f:
                        content = f.read()
                        if content.strip():
                            found_results.append({
                                "job_id": job_id,
                                "result_file": result_file,
                                "content": content.strip()
                            })
                except:
                    pass
        return found_results
    
    def display_status(self):
        """显示状态面板"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🚀 Bitcoin Puzzle 135 实时监控面板")
        print("=" * 80)
        print(f"启动时间: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"运行时长: {datetime.now() - self.start_time}")
        print()
        
        # GPU状态
        gpus = self.get_gpu_status()
        if gpus:
            print("🖥️  GPU状态:")
            for gpu in gpus:
                memory_percent = (gpu["memory_used"] / gpu["memory_total"]) * 100
                print(f"   GPU{gpu['id']}: {gpu['name']}")
                print(f"     利用率: {gpu['utilization']}% | 显存: {memory_percent:.1f}% ({gpu['memory_used']}/{gpu['memory_total']}MB) | 温度: {gpu['temperature']}°C")
            print()
        
        # 任务状态统计
        status_counts = defaultdict(int)
        total_progress = 0
        total_speed = 0
        active_jobs = 0
        
        print("📊 任务状态:")
        print(f"{'任务ID':<25} {'GPU':<4} {'状态':<12} {'进度':<8} {'速度':<12} {'最后更新':<20}")
        print("-" * 80)
        
        for job_id, job in self.jobs.items():
            log_file = job["kangaroo_params"]["log_file"]
            log_info = self.parse_log_file(log_file)
            
            status_counts[log_info["status"]] += 1
            total_progress += log_info["progress"]
            
            if log_info["status"] == "running":
                total_speed += log_info["speed"]
                active_jobs += 1
            
            # 格式化显示
            job_display = job_id[:24]
            gpu_id = f"GPU{job['gpu_id']}"
            status = log_info["status"]
            progress = f"{log_info['progress']:.1f}%"
            speed = f"{log_info['speed']:.1f}MK/s" if log_info["speed"] > 0 else "-"
            last_update = log_info.get("last_update", "").strftime("%H:%M:%S") if log_info.get("last_update") else "-"
            
            print(f"{job_display:<25} {gpu_id:<4} {status:<12} {progress:<8} {speed:<12} {last_update:<20}")
        
        print("-" * 80)
        
        # 总体统计
        total_jobs = len(self.jobs)
        avg_progress = total_progress / total_jobs if total_jobs > 0 else 0
        
        print(f"\n📈 总体统计:")
        print(f"   总任务数: {total_jobs}")
        print(f"   平均进度: {avg_progress:.1f}%")
        print(f"   总速度: {total_speed:.1f} MK/s")
        print(f"   活跃任务: {active_jobs}")
        
        for status, count in status_counts.items():
            print(f"   {status}: {count}")
        
        # 检查结果
        results = self.check_results()
        if results:
            print(f"\n🎉 发现结果!")
            for result in results:
                print(f"   任务: {result['job_id']}")
                print(f"   文件: {result['result_file']}")
                print(f"   内容: {result['content'][:100]}...")
        
        # 预计完成时间
        if avg_progress > 0 and avg_progress < 100:
            elapsed = datetime.now() - self.start_time
            estimated_total = elapsed / (avg_progress / 100)
            eta = self.start_time + estimated_total
            print(f"\n⏰ 预计完成时间: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n🔄 刷新时间: {datetime.now().strftime('%H:%M:%S')} (每30秒自动刷新)")
        print("按 Ctrl+C 退出监控")
    
    def run(self, refresh_interval=30):
        """运行监控"""
        try:
            while True:
                self.display_status()
                
                # 检查是否有结果
                results = self.check_results()
                if results:
                    print(f"\n🎉 发现私钥！停止监控...")
                    for result in results:
                        print(f"任务 {result['job_id']} 找到结果:")
                        print(result['content'])
                    break
                
                time.sleep(refresh_interval)
                
        except KeyboardInterrupt:
            print(f"\n\n👋 监控已停止")

def main():
    if len(sys.argv) > 1:
        jobs_file = sys.argv[1]
    else:
        jobs_file = "jobs.json"
    
    if not os.path.exists(jobs_file):
        print(f"❌ 任务配置文件不存在: {jobs_file}")
        print("请先运行 GLV 预过滤和任务生成")
        return 1
    
    monitor = Puzzle135Monitor(jobs_file)
    print(f"🔍 开始监控 {len(monitor.jobs)} 个任务...")
    monitor.run()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
