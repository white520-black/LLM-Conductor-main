import json
import os
import datetime  # 完整导入datetime模块
from monitor.config import PATH_CONFIG


class ReportGenerator:
    def __init__(self, collected_data, analysis_results, plot_paths):
        self.collected_data = collected_data
        self.analysis_results = analysis_results
        self.plot_paths = plot_paths
        self.report_path = os.path.join(PATH_CONFIG["results"], "analysis_report.html")
        self.txt_report_path = os.path.join(PATH_CONFIG["results"], "analysis_report.txt")

    def _generate_summary(self):
        """生成监控数据汇总（修正datetime调用）"""
        if not self.collected_data:
            return {"total_time": 0, "cpu_avg": 0, "mem_avg": 0, "abnormal_count": 0}

        # 时间范围（修正：用datetime.datetime.fromtimestamp）
        start_ts = self.collected_data[0]["system"]["timestamp"]
        end_ts = self.collected_data[-1]["system"]["timestamp"]
        total_time = round(end_ts - start_ts, 1)

        # 资源平均值
        cpu_avg = round(sum(d["system"]["cpu_usage"] for d in self.collected_data) / len(self.collected_data), 1)
        mem_avg = round(sum(d["system"]["mem_usage"] for d in self.collected_data) / len(self.collected_data), 1)

        # GPU平均值（适配无GPU）
        has_gpu = len(self.collected_data[0]["gpu"]) > 0
        gpu0_avg = 0.0
        gpu1_avg = 0.0
        if has_gpu:
            gpu0_avg = round(sum(d["gpu"][0]["gpu_usage"] for d in self.collected_data if len(d["gpu"]) > 0) / len(
                self.collected_data), 1)
            if len(self.collected_data[0]["gpu"]) > 1:
                gpu1_avg = round(sum(d["gpu"][1]["gpu_usage"] for d in self.collected_data if len(d["gpu"]) > 1) / len(
                    self.collected_data), 1)

        # 异常统计
        resource_abnormal_count = sum(1 for res in self.analysis_results if res["ml_anomalies"]["resource_abnormal"])
        log_abnormal_count = sum(1 for res in self.analysis_results if res["ml_anomalies"]["abnormal_logs"])

        # 端口连接数
        port_6006_avg = round(sum(next(p["established_connections"] for p in d["network"] if p["port"] == 6006) for d in
                                  self.collected_data) / len(self.collected_data), 1)

        return {
            "start_time": datetime.datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": datetime.datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S"),
            "total_time": total_time,
            "cpu_avg": cpu_avg,
            "mem_avg": mem_avg,
            "has_gpu": has_gpu,
            "gpu0_avg": gpu0_avg,
            "gpu1_avg": gpu1_avg,
            "resource_abnormal_count": resource_abnormal_count,
            "log_abnormal_count": log_abnormal_count,
            "port_6006_avg": port_6006_avg,
            "total_log_count": sum(len(d["logs"]) for d in self.collected_data)
        }

    def _generate_llm_analysis_summary(self):
        """汇总LLM分析结果"""
        llm_summaries = [res["llm_analysis"] for res in self.analysis_results if res["llm_analysis"]]
        # 取最后5条LLM分析（避免报告过长）
        return llm_summaries[-5:] if len(llm_summaries) > 5 else llm_summaries

    def generate_html_report(self):
        """生成HTML报告（可视化友好）"""
        summary = self._generate_summary()
        llm_summaries = self._generate_llm_analysis_summary()

        # HTML模板（嵌入图表和汇总数据）
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>LLM-Sentry 监控分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .chart-container {{ margin: 30px 0; text-align: center; }}
        .chart-img {{ max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 8px; }}
        .llm-analysis {{ background: #e3f2fd; padding: 15px; border-radius: 8px; margin: 10px 0; }}
        .warning {{ color: #e67e22; }}
        .error {{ color: #e74c3c; }}
        .success {{ color: #27ae60; }}
    </style>
</head>
<body>
    <h1>📊 LLM-Sentry 监控分析报告</h1>
    <p><strong>报告生成时间：</strong>{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <!-- 监控汇总 -->
    <div class="summary-card">
        <h2>一、监控核心汇总</h2>
        <p><strong>监控时段：</strong>{summary["start_time"]} ~ {summary["end_time"]}（总计 {summary["total_time"]} 秒）</p>
        <p><strong>系统资源平均值：</strong>CPU {summary["cpu_avg"]}% | 内存 {summary["mem_avg"]}%</p>
        {f'<p><strong>GPU资源平均值：</strong>GPU0 {summary["gpu0_avg"]}% | GPU1 {summary["gpu1_avg"]}%</p>' if summary["has_gpu"] else '<p><strong>GPU资源：</strong>未启用</p>'}
        <p><strong>端口6006平均连接数：</strong>{summary["port_6006_avg"]}</p>
        <p><strong>日志总条数：</strong>{summary["total_log_count"]}</p>
        <p><strong>异常事件统计：</strong>资源异常 {summary["resource_abnormal_count"]} 次 | 日志异常 {summary["log_abnormal_count"]} 次</p>
    </div>

    <!-- 可视化图表 -->
    <h2>二、可视化分析图表</h2>
    <div class="chart-container">
        <h3>资源使用率趋势图</h3>
        <img src="{os.path.basename(self.plot_paths.get('resource_trend', ''))}" class="chart-img" alt="资源趋势图">
    </div>
    <div class="chart-container">
        <h3>异常事件统计图</h3>
        <img src="{os.path.basename(self.plot_paths.get('abnormal_events', ''))}" class="chart-img" alt="异常事件统计图">
    </div>
    <div class="chart-container">
        <h3>进程资源占用趋势</h3>
        <img src="{os.path.basename(self.plot_paths.get('process_status', ''))}" class="chart-img" alt="进程状态图">
    </div>

    <!-- LLM深度分析 -->
    <h2>三、LLM深度分析结果</h2>
    {''.join([f'<div class="llm-analysis"><strong>分析片段 {i + 1}：</strong>{summary}</div>' for i, summary in enumerate(llm_summaries)])}

    <!-- 风险提示 -->
    <h2>四、风险提示与建议</h2>
    <div class="warning">
        <p><strong>⚠️ 风险提示：</strong>
            {f'共检测到 {summary["resource_abnormal_count"]} 次资源异常，建议检查CPU/GPU负载是否过高，是否存在资源泄露。' if summary["resource_abnormal_count"] > 0 else '未检测到资源异常，系统资源运行稳定。'}
            {f'共检测到 {summary["log_abnormal_count"]} 次日志异常，建议查看原始日志排查潜在问题。' if summary["log_abnormal_count"] > 0 else '未检测到日志异常（或未启用日志异常检测）。'}
        </p>
    </div>
    <div class="success">
        <p><strong>✅ 优化建议：</strong></p>
        <ul>
            <li>若CPU/GPU使用率持续过高，建议优化模型推理批处理大小或限制资源使用上限；</li>
            <li>若端口连接数异常，建议检查客户端请求是否合理，是否存在连接泄露；</li>
            <li>定期备份监控报告和原始数据，便于后续问题复盘；</li>
            <li>若需要更精准的异常检测，可调整ML模型参数（如Isolation Forest的contamination阈值）。</li>
        </ul>
    </div>
</body>
</html>
        """

        # 写入HTML文件
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(html_template)

        # 修复：调用时传入 summary 和 llm_summaries 参数
        self._generate_txt_report(summary, llm_summaries)

        return self.report_path

    # 修复：方法定义添加 summary 和 llm_summaries 参数
    def _generate_txt_report(self, summary, llm_summaries):
        """生成TXT版本报告（简洁易读）"""
        txt_content = f"""
===================================== LLM-Sentry 监控分析报告 =====================================
报告生成时间：{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
===================================================================================================
一、监控核心汇总
- 监控时段：{summary["start_time"]} ~ {summary["end_time"]}（总计 {summary["total_time"]} 秒）
- 系统资源平均值：CPU {summary["cpu_avg"]}% | 内存 {summary["mem_avg"]}%
{f'- GPU资源平均值：GPU0 {summary["gpu0_avg"]}% | GPU1 {summary["gpu1_avg"]}%' if summary["has_gpu"] else '- GPU资源：未启用'}
- 端口6006平均连接数：{summary["port_6006_avg"]}
- 日志总条数：{summary["total_log_count"]}
- 异常事件统计：资源异常 {summary["resource_abnormal_count"]} 次 | 日志异常 {summary["log_abnormal_count"]} 次

二、LLM深度分析结果
{''.join([f'\n【分析片段 {i + 1}】\n{summary}\n' for i, summary in enumerate(llm_summaries)])}

三、风险提示与建议
⚠️  风险提示：
{f'共检测到 {summary["resource_abnormal_count"]} 次资源异常，建议检查CPU/GPU负载是否过高，是否存在资源泄露。' if summary["resource_abnormal_count"] > 0 else '未检测到资源异常，系统资源运行稳定。'}
{f'共检测到 {summary["log_abnormal_count"]} 次日志异常，建议查看原始日志排查潜在问题。' if summary["log_abnormal_count"] > 0 else '未检测到日志异常（或未启用日志异常检测）。'}

✅ 优化建议：
1. 若CPU/GPU使用率持续过高，建议优化模型推理批处理大小或限制资源使用上限；
2. 若端口连接数异常，建议检查客户端请求是否合理，是否存在连接泄露；
3. 定期备份监控报告和原始数据，便于后续问题复盘；
4. 若需要更精准的异常检测，可调整ML模型参数（如Isolation Forest的contamination阈值）。
===================================================================================================
        """

        with open(self.txt_report_path, "w", encoding="utf-8") as f:
            f.write(txt_content.strip())


# 兼容旧版本调用（若有）
if __name__ == "__main__":
    ReportGenerator([], [], {}).generate_html_report()