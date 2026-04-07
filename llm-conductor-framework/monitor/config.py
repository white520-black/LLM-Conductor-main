import os

# 监控基础配置
MONITOR_CONFIG = {
    "collect_interval": 1,          # 数据收集频率（秒/次）
    "baseline_duration": 30,        # 基线数据收集时间（用于ML模型训练）
    "log_paths": [
        "/home/xukai/SecGPT-IsolateGPT-AE/results/case_log.txt",
        "/home/xukai/SecGPT-IsolateGPT-AE/results/isolategpt_case1.txt",
        "/home/xukai/SecGPT-IsolateGPT-AE/results/isolategpt_case2.txt",
        "/home/xukai/SecGPT-IsolateGPT-AE/results/isolategpt_case3.txt",
        "/home/xukai/SecGPT-IsolateGPT-AE/results/isolategpt_case4.txt",
        # 原项目日志路径（支持多个，零入侵读取）
    ],
    "process_name": None,           # 原项目进程名（可选，用于进程状态监控）
    "process_pid": None,            # 原项目PID（可选，优先级高于进程名）
    "monitor_ports": [6006],        # 监控端口（vllm端口+原项目端口）
    "gpu_device_ids": [0, 1]        # GPU设备ID（适配tensor-parallel-size=2）
}

# 千问模型配置（用户指定参数）
LLM_CONFIG = {
    "base_url": "http://127.0.0.1:6006/v1",
    "api_key": "xxx",
    "model_name": "Qwen2-72B-Instruct",
    "max_tokens": 2048,
    "temperature": 0.3
}

# 路径配置（固定）
PATH_CONFIG = {
    "raw_data": "./data/raw_data.json",  # 原始数据存储路径
    "results": "./results",              # 可视化+报告输出路径
    "logs": "./logs/llm_sentry.log",     # 监控模块自身日志路径
    "status": "./.status"                # 监控状态文件路径
}

# 创建必要目录（含 data 目录，避免写入失败）
for dir_path in [
    PATH_CONFIG["results"],
    os.path.dirname(PATH_CONFIG["logs"]),
    os.path.dirname(PATH_CONFIG["raw_data"])  # 新增：自动创建 data 目录
]:
    os.makedirs(dir_path, exist_ok=True)