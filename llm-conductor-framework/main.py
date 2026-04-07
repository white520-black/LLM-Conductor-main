import argparse
import time
import json
import logging
from datetime import datetime
from monitor.collector import DataCollector
from analyzer.analyzer_core import AnalysisCore
from visualizer.plotter import Visualizer
from visualizer.report_generator import ReportGenerator
from monitor.config import PATH_CONFIG, MONITOR_CONFIG
import os

# 配置日志
logging.basicConfig(
    filename=PATH_CONFIG["logs"],
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_status():
    """加载监控状态"""
    if not os.path.exists(PATH_CONFIG["status"]):
        return {"status": "stopped", "start_time": None, "duration": 0}
    with open(PATH_CONFIG["status"], "r") as f:
        return json.load(f)


def save_status(status):
    """保存监控状态"""
    with open(PATH_CONFIG["status"], "w") as f:
        json.dump(status, f)


def start_monitor(duration):
    """启动监控（含原始数据持久化）"""
    status = load_status()
    if status["status"] == "running":
        print("❌ 监控已在运行中！")
        logging.info("Monitor already running")
        return

    # 更新状态
    status = {
        "status": "running",
        "start_time": time.time(),
        "duration": duration,
        "end_time": time.time() + duration
    }
    save_status(status)
    print(f"✅ 监控已启动，持续时间：{duration}秒（{duration / 60:.1f}分钟）")
    logging.info(f"Monitor started, duration: {duration}s")

    # 初始化组件
    collector = DataCollector()
    analyzer = AnalysisCore()
    collected_data = []
    analysis_results = []

    try:
        # 1. 启动日志监听（零入侵）
        collector.start_log_monitor()

        # 2. 收集基线数据并训练ML模型
        print("🔍 正在收集基线数据（30秒）...")
        baseline_count = 0
        while baseline_count < MONITOR_CONFIG["baseline_duration"]:
            data = collector.collect()
            collected_data.append(data)
            analyzer.collect_baseline(data)
            baseline_count += MONITOR_CONFIG["collect_interval"]
            time.sleep(MONITOR_CONFIG["collect_interval"])

        # 训练模型
        analyzer.train_models()
        print("✅ ML模型训练完成，开始实时监控...")

        # 3. 实时监控+分析
        while time.time() < status["end_time"]:
            if load_status()["status"] == "stopped":
                print("🛑 监控被手动停止")
                logging.info("Monitor stopped manually")
                break

            # 收集数据
            data = collector.collect()
            collected_data.append(data)

            # 分析数据
            analysis = analyzer.analyze(data)
            analysis_results.append(analysis)

            # 打印实时异常
            if analysis["ml_anomalies"]["resource_abnormal"]:
                print(f"⚠️ [{datetime.now().strftime('%H:%M:%S')}] 检测到资源异常！")
                logging.warning("Resource anomaly detected")
            if analysis["ml_anomalies"]["abnormal_logs"]:
                print(
                    f"⚠️ [{datetime.now().strftime('%H:%M:%S')}] 检测到{len(analysis['ml_anomalies']['abnormal_logs'])}条异常日志")
                logging.warning(f"Abnormal logs detected: {len(analysis['ml_anomalies']['abnormal_logs'])}")

            time.sleep(MONITOR_CONFIG["collect_interval"])

        # 4. 监控结束，生成可视化和报告
        print("📊 监控结束，正在生成分析结果...")
        visualizer = Visualizer(collected_data, analysis_results)
        plot_paths = visualizer.generate_all_plots()
        report_path = ReportGenerator(collected_data, analysis_results, plot_paths).generate_html_report()

        print(f"🎉 分析完成！结果已保存至：{PATH_CONFIG['results']}")
        print(f"📄 分析报告：{report_path}")
        logging.info("Monitor completed, results generated")

        # 关键修复：将原始数据写入 raw_data.json
        try:
            # 确保 data 目录存在（避免目录不存在报错）
            data_dir = os.path.dirname(PATH_CONFIG["raw_data"])
            if not os.path.exists(data_dir):
                os.makedirs(data_dir, exist_ok=True)
                print(f"✅ 已创建 data 目录：{data_dir}")

            # 写入收集到的所有原始数据（基线+实时）
            with open(PATH_CONFIG["raw_data"], "w", encoding="utf-8") as f:
                json.dump(collected_data, f, ensure_ascii=False, indent=2)
            print(f"✅ 原始数据已保存至：{PATH_CONFIG['raw_data']}")
            logging.info(f"Raw data saved to: {PATH_CONFIG['raw_data']}")
        except Exception as e:
            print(f"⚠️  写入 raw_data.json 失败：{str(e)}")
            logging.error(f"Failed to write raw_data.json: {str(e)}")

    except Exception as e:
        print(f"❌ 监控异常：{str(e)}")
        logging.error(f"Monitor error: {str(e)}")
    finally:
        # 清理资源
        collector.stop_log_monitor()
        # 重置状态
        save_status({"status": "stopped", "start_time": None, "duration": 0, "end_time": None})


def stop_monitor():
    """停止监控"""
    status = load_status()
    if status["status"] == "stopped":
        print("❌ 监控未在运行中！")
        return

    status["status"] = "stopped"
    save_status(status)
    print("✅ 监控已停止")
    logging.info("Monitor stopped manually")


def status_monitor():
    """查看监控状态"""
    status = load_status()
    if status["status"] == "running":
        elapsed = time.time() - status["start_time"]
        remaining = status["end_time"] - time.time()
        print(f"📋 监控状态：运行中")
        print(f"   开始时间：{datetime.fromtimestamp(status['start_time']).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   已运行：{elapsed:.1f}秒")
        print(f"   剩余时间：{max(0, remaining):.1f}秒")
    else:
        print("📋 监控状态：已停止")


def main():
    parser = argparse.ArgumentParser(description="LLM-Sentry 监控模块（零入侵、多算法分析）")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 启动命令
    start_parser = subparsers.add_parser("start", help="启动监控")
    start_parser.add_argument("--duration", type=int, required=True, help="监控持续时间（秒）")

    # 停止命令
    subparsers.add_parser("stop", help="停止监控")

    # 状态命令
    subparsers.add_parser("status", help="查看监控状态")

    args = parser.parse_args()

    if args.command == "start":
        start_monitor(args.duration)
    elif args.command == "stop":
        stop_monitor()
    elif args.command == "status":
        status_monitor()


if __name__ == "__main__":
    main()