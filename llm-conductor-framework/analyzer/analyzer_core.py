import numpy as np
from analyzer.ml_algorithms import MLAnalyzer
from analyzer.llm_client import LLMClient


class AnalysisCore:
    def __init__(self):
        self.ml_analyzer = MLAnalyzer()
        self.llm_client = LLMClient()
        self.baseline_data = []  # 基线数据（用于训练ML模型）
        self.collected_data = []  # 所有收集到的数据
        self.has_gpu = False  # 是否有GPU数据
        self.log_model_available = False  # 日志模型是否可用

    def collect_baseline(self, data):
        """收集基线数据（用于ML模型训练）"""
        self.baseline_data.append(data)
        # 检测是否有GPU数据
        if not self.has_gpu and len(data["gpu"]) > 0:
            self.has_gpu = True
        return len(self.baseline_data)

    def train_models(self):
        """训练所有ML模型（适配无日志场景）"""
        # 训练资源异常检测模型
        self.ml_analyzer.train_resource_model(self.baseline_data)
        print("✅ 资源异常检测模型训练完成")

        # 训练日志异常检测模型（兼容无日志）
        baseline_logs = [d["logs"] for d in self.baseline_data if d["logs"]]
        self.ml_analyzer.train_log_model(baseline_logs)
        self.log_model_available = self.ml_analyzer.log_model_trained  # 标记日志模型是否可用

    def analyze(self, data):
        """实时分析单条监控数据"""
        self.collected_data.append(data)

        # 1. ML算法异常检测
        resource_abnormal = self.ml_analyzer.detect_resource_anomaly(data)
        # 日志模型不可用时，跳过日志异常检测
        abnormal_logs = self.ml_analyzer.detect_log_anomaly(data["logs"]) if self.log_model_available else []

        ml_anomalies = {
            "resource_abnormal": resource_abnormal,
            "abnormal_logs": abnormal_logs
        }

        # 2. 生成监控摘要（用于LLM分析）
        monitor_summary = self._generate_summary()

        # 3. LLM深度分析
        llm_analysis = self.llm_client.analyze_with_llm(monitor_summary, ml_anomalies)

        return {
            "ml_anomalies": ml_anomalies,
            "llm_analysis": llm_analysis,
            "monitor_summary": monitor_summary
        }

    def _generate_summary(self):
        """生成监控数据摘要（适配无GPU/无日志）"""
        recent_data = self.collected_data[-60:]  # 取最近60条（1分钟，按1秒/条）
        if not recent_data:
            return {}

        # 系统资源平均
        cpu_avg = np.mean([d["system"]["cpu_usage"] for d in recent_data])
        mem_avg = np.mean([d["system"]["mem_usage"] for d in recent_data])

        # 初始化监控摘要
        monitor_summary = {
            "cpu_avg": cpu_avg, "mem_avg": mem_avg,
        }

        # 有GPU时添加GPU指标
        if self.has_gpu:
            gpu0_usage = np.mean([d["gpu"][0]["gpu_usage"] for d in recent_data if len(d["gpu"]) > 0])
            gpu0_mem = np.mean([d["gpu"][0]["mem_used"] for d in recent_data if len(d["gpu"]) > 0])
            gpu1_usage = np.mean([d["gpu"][1]["gpu_usage"] for d in recent_data if len(d["gpu"]) > 1])
            gpu1_mem = np.mean([d["gpu"][1]["mem_used"] for d in recent_data if len(d["gpu"]) > 1])
            monitor_summary.update({
                "gpu0_usage": gpu0_usage, "gpu0_mem": gpu0_mem,
                "gpu1_usage": gpu1_usage, "gpu1_mem": gpu1_mem,
            })

        # 进程状态
        process_status = recent_data[-1]["process"]["status"]
        process_cpu = np.mean([d["process"]["cpu_usage"] for d in recent_data])

        # 网络连接（端口6006）
        port_6006_conn = np.mean([
            next(p["established_connections"] for p in d["network"] if p["port"] == 6006)
            for d in recent_data
        ])

        # 日志统计（区分是否有日志模型）
        log_count = sum(len(d["logs"]) for d in recent_data)
        monitor_summary.update({
            "process_status": process_status, "process_cpu": process_cpu,
            "port_6006_conn": port_6006_conn,
            "log_count": log_count,
            "log_model_available": self.log_model_available  # 告诉LLM是否有日志异常检测
        })

        return monitor_summary