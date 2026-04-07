from openai import OpenAI
from monitor.config import LLM_CONFIG


class LLMClient:
    def __init__(self):
        # 初始化千问模型客户端（vllm服务）
        self.client = OpenAI(
            base_url=LLM_CONFIG["base_url"],
            api_key=LLM_CONFIG["api_key"],
            timeout=30
        )

    def analyze_with_llm(self, monitor_summary, ml_anomalies):
        """调用千问模型分析监控数据和ML异常结果（适配无日志模型）"""
        # 构造GPU信息文本（适配无GPU）
        gpu_info = ""
        if "gpu0_usage" in monitor_summary:
            gpu_info = f"GPU0使用率{monitor_summary['gpu0_usage']:.1f}%（显存使用{monitor_summary['gpu0_mem']:.2f}GB），GPU1使用率{monitor_summary['gpu1_usage']:.1f}%（显存使用{monitor_summary['gpu1_mem']:.2f}GB）"
        else:
            gpu_info = "无GPU监控数据"

        # 构造日志信息文本（适配无日志模型）
        log_info = f"日志条数：{monitor_summary['log_count']}条"
        if not monitor_summary.get("log_model_available", False):
            log_info += "（日志不足，未启用日志异常检测）"

        # 构造异常日志文本
        abnormal_logs_text = ml_anomalies["abnormal_logs"] if ml_anomalies["abnormal_logs"] else "无"
        if not monitor_summary.get("log_model_available", False):
            abnormal_logs_text = "未启用日志异常检测（日志不足）"

        # 最终Prompt
        prompt = f"""
        你是LLM-Sentry监控分析助手，需要分析以下项目监控数据和异常检测结果：

        一、监控数据摘要（最近1分钟）：
        1. 系统资源：CPU使用率{monitor_summary['cpu_avg']:.1f}%，内存使用率{monitor_summary['mem_avg']:.1f}%
        2. GPU资源：{gpu_info}
        3. 进程状态：{monitor_summary['process_status']}，CPU占用{monitor_summary['process_cpu']:.1f}%
        4. 网络连接：端口6006活跃连接数{monitor_summary['port_6006_conn']}
        5. {log_info}

        二、机器学习异常检测结果：
        1. 资源异常：{'存在' if ml_anomalies['resource_abnormal'] else '无'}
        2. 异常日志：{abnormal_logs_text}

        请完成以下分析：
        1. 判断当前系统是否存在异常风险（如资源过载、进程异常、端口连接异常等）
        2. 分析异常原因（如果存在异常）
        3. 给出优化建议（如调整资源分配、优化进程、检查端口等）
        4. 总结当前系统运行状态

        要求：分析简洁明了，重点突出，建议具有可操作性；若未启用日志异常检测，无需提及日志相关的异常分析。
        """

        try:
            response = self.client.chat.completions.create(
                model=LLM_CONFIG["model_name"],
                messages=[{"role": "user", "content": prompt}],
                max_tokens=LLM_CONFIG["max_tokens"],
                temperature=LLM_CONFIG["temperature"]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM调用失败：{str(e)}（请确保vllm服务已启动，参数配置正确）"