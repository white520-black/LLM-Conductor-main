import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime  # 确保完整导入
from monitor.config import PATH_CONFIG

# 解决中文显示问题 + 基础样式配置
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['figure.autolayout'] = True  # 自动调整布局，避免tight_layout警告


class Visualizer:
    def __init__(self, collected_data, analysis_results):
        self.df = self._convert_to_df(collected_data)
        self.analysis_results = analysis_results
        self.save_path = PATH_CONFIG["results"]
        self.has_gpu = len(collected_data) > 0 and len(collected_data[0]["gpu"]) > 0
        # 创建结果目录
        import os
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def _convert_to_df(self, collected_data):
        """将原始数据转为DataFrame（适配无GPU）"""
        rows = []
        for data in collected_data:
            # 修正：用datetime.datetime.fromtimestamp
            ts = datetime.datetime.fromtimestamp(data["system"]["timestamp"])
            row = {
                "timestamp": ts,
                "cpu_usage": data["system"]["cpu_usage"],
                "mem_usage": data["system"]["mem_usage"],
                "process_cpu": data["process"]["cpu_usage"],
                "process_mem": data["process"]["mem_usage"],
                "port_6006_conn": next(p["established_connections"] for p in data["network"] if p["port"] == 6006),
                "log_count": len(data["logs"])
            }
            if data["gpu"]:
                row["gpu0_usage"] = data["gpu"][0]["gpu_usage"] if len(data["gpu"]) > 0 else 0.0
                row["gpu1_usage"] = data["gpu"][1]["gpu_usage"] if len(data["gpu"]) > 1 else 0.0
                row["gpu0_mem"] = data["gpu"][0]["mem_used"] if len(data["gpu"]) > 0 else 0.0
                row["gpu1_mem"] = data["gpu"][1]["mem_used"] if len(data["gpu"]) > 1 else 0.0
            rows.append(row)
        return pd.DataFrame(rows)

    def plot_resource_trend(self):
        """资源使用率趋势图（适配无GPU）"""
        if self.has_gpu:
            fig, axes = plt.subplots(3, 1, sharex=True, figsize=(15, 12))
        else:
            fig, axes = plt.subplots(2, 1, sharex=True, figsize=(15, 8))

        # CPU趋势
        sns.lineplot(data=self.df, x="timestamp", y="cpu_usage", ax=axes[0], color="#1f77b4", linewidth=2)
        axes[0].set_title("CPU Usage Trend (%)", fontsize=14, fontweight='bold')
        axes[0].axhline(y=80, color="red", linestyle="--", alpha=0.7, label="Threshold (80%)")
        axes[0].set_ylabel("CPU Usage (%)", fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 内存趋势
        sns.lineplot(data=self.df, x="timestamp", y="mem_usage", ax=axes[1], color="#ff7f0e", linewidth=2)
        axes[1].set_title("Memory Usage Trend (%)", fontsize=14, fontweight='bold')
        axes[1].axhline(y=85, color="red", linestyle="--", alpha=0.7, label="Threshold (85%)")
        axes[1].set_ylabel("Memory Usage (%)", fontsize=12)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # GPU趋势（有GPU时）
        if self.has_gpu:
            sns.lineplot(data=self.df, x="timestamp", y="gpu0_usage", ax=axes[2], color="#2ca02c", label="GPU 0",
                         linewidth=2)
            if "gpu1_usage" in self.df.columns and self.df["gpu1_usage"].max() > 0:
                sns.lineplot(data=self.df, x="timestamp", y="gpu1_usage", ax=axes[2], color="#d62728", label="GPU 1",
                             linewidth=2)
            axes[2].set_title("GPU Usage Trend (%)", fontsize=14, fontweight='bold')
            axes[2].axhline(y=95, color="red", linestyle="--", alpha=0.7, label="Threshold (95%)")
            axes[2].set_ylabel("GPU Usage (%)", fontsize=12)
            axes[2].set_xlabel("Time", fontsize=12)
            axes[2].legend(fontsize=10)
            axes[2].grid(True, alpha=0.3)
        else:
            axes[1].set_xlabel("Time", fontsize=12)

        plt.xticks(rotation=45)
        plt.savefig(f"{self.save_path}/resource_trend.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 资源趋势图已保存至：{self.save_path}/resource_trend.png")

    def plot_abnormal_events(self):
        """异常事件统计图表（修复ylim冲突+布局警告）"""
        # 统计异常次数
        resource_abnormal_count = sum(1 for res in self.analysis_results if res["ml_anomalies"]["resource_abnormal"])
        log_abnormal_count = sum(1 for res in self.analysis_results if res["ml_anomalies"]["abnormal_logs"])

        categories = ["Resource Anomalies", "Log Anomalies"]
        counts = [resource_abnormal_count, log_abnormal_count]
        colors = ["#ff6b6b", "#4ecdc4"]

        fig, ax = plt.subplots(figsize=(10, 6))
        # 修复seaborn警告：hue+legend=False
        sns.barplot(
            x=categories,
            y=counts,
            ax=ax,
            palette=colors,
            hue=categories,
            legend=False
        )

        # 修复ylim冲突：当counts全为0时，设置y轴范围为0-1
        max_count = max(counts) if counts else 0
        y_max = max_count * 1.2 if max_count > 0 else 1
        ax.set_ylim(0, y_max)

        # 数值标签（避免超出y轴）
        for i, count in enumerate(counts):
            ax.text(i, count + y_max * 0.02, str(count), ha="center", va="bottom", fontsize=12, fontweight='bold')

        ax.set_title("Abnormal Events Statistics", fontsize=14, fontweight='bold')
        ax.set_ylabel("Number of Abnormal Events", fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        plt.savefig(f"{self.save_path}/abnormal_events.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 异常事件统计图已保存至：{self.save_path}/abnormal_events.png")

    def plot_process_status(self):
        """进程CPU/内存占用趋势"""
        fig, ax = plt.subplots(figsize=(15, 6))
        sns.lineplot(data=self.df, x="timestamp", y="process_cpu", ax=ax, color="#9b59b6", label="Process CPU Usage",
                     linewidth=2)
        sns.lineplot(data=self.df, x="timestamp", y="process_mem", ax=ax, color="#e67e22", label="Process Memory Usage",
                     linewidth=2)

        ax.set_title("Process Resource Usage Trend", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel("Usage (%)", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.savefig(f"{self.save_path}/process_status.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ 进程状态图已保存至：{self.save_path}/process_status.png")

    def generate_all_plots(self):
        """生成所有可视化图表"""
        try:
            self.plot_resource_trend()
            self.plot_abnormal_events()
            self.plot_process_status()
            return {
                "resource_trend": f"{self.save_path}/resource_trend.png",
                "abnormal_events": f"{self.save_path}/abnormal_events.png",
                "process_status": f"{self.save_path}/process_status.png"
            }
        except Exception as e:
            print(f"⚠️  生成可视化图表时出错：{e}")
            return {"error": str(e)}