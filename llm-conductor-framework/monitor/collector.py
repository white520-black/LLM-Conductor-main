import time
import os
import json
import psutil
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from monitor.config import MONITOR_CONFIG, PATH_CONFIG

# 容错：尝试导入pynvml，失败则禁用GPU监控
try:
    import pynvml

    PYNVMl_AVAILABLE = True
except ImportError:
    PYNVMl_AVAILABLE = False
    print("⚠️  未找到 pynvml 模块，禁用 GPU 监控功能")


class LogHandler(FileSystemEventHandler):
    """日志文件监听处理器（零入侵读取）"""

    def __init__(self):
        self.latest_logs = []

    def on_modified(self, event):
        if not event.is_directory:
            try:
                with open(event.src_path, "r", encoding="utf-8") as f:
                    # 读取新增日志（最后10行，避免读取过大文件）
                    lines = f.readlines()[-10:]
                    self.latest_logs.extend([line.strip() for line in lines if line.strip()])
            except Exception as e:
                print(f"⚠️  读取日志文件失败：{event.src_path}，错误：{e}")


class DataCollector:
    def __init__(self):
        self.log_handler = LogHandler()  # 初始化日志处理器
        self.log_observer = Observer()  # 初始化日志监听观察者
        self.gpu_handles = []  # GPU设备句柄列表
        self._init_gpu()  # 初始化GPU监控（容错版）

    def _init_gpu(self):
        """初始化GPU监控（容错版：失败不崩溃，仅禁用GPU监控）"""
        if not PYNVMl_AVAILABLE:
            return

        try:
            pynvml.nvmlInit()
            # 过滤不存在的GPU设备ID
            total_gpus = pynvml.nvmlDeviceGetCount()
            valid_gpu_ids = [idx for idx in MONITOR_CONFIG["gpu_device_ids"] if 0 <= idx < total_gpus]
            self.gpu_handles = [pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in valid_gpu_ids]

            if not self.gpu_handles:
                print(f"⚠️  未找到有效GPU设备（配置的GPU ID：{MONITOR_CONFIG['gpu_device_ids']}，实际可用：{total_gpus}个）")
            else:
                print(f"✅ GPU监控初始化成功，监控设备：{valid_gpu_ids}")
        except Exception as e:
            print(f"⚠️  GPU初始化失败：{e}，禁用GPU监控功能")
            self.gpu_handles = []

    def _collect_system(self):
        """收集系统资源（CPU/内存）"""
        return {
            "cpu_usage": psutil.cpu_percent(interval=0.1),
            "mem_usage": psutil.virtual_memory().percent,
            "timestamp": time.time()
        }

    def _collect_gpu(self):
        """收集GPU资源（适配vllm tensor-parallel-size=2，容错版）"""
        gpu_data = []
        if not PYNVMl_AVAILABLE or not self.gpu_handles:
            return gpu_data

        try:
            for handle in self.gpu_handles:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_data.append({
                    "gpu_id": pynvml.nvmlDeviceGetIndex(handle),
                    "gpu_usage": util.gpu,
                    "mem_used": round(mem.used / 1024 / 1024 / 1024, 2),  # 转为GB并保留2位小数
                    "mem_total": round(mem.total / 1024 / 1024 / 1024, 2)
                })
        except Exception as e:
            print(f"⚠️  GPU数据收集失败：{e}")
        return gpu_data

    def _collect_process(self):
        """收集原项目进程状态（零入侵）"""
        process_data = {"status": "stopped", "cpu_usage": 0.0, "mem_usage": 0.0, "threads": 0}
        try:
            # 优先通过PID获取，无则通过进程名
            target_p = None
            if MONITOR_CONFIG["process_pid"]:
                try:
                    target_p = psutil.Process(MONITOR_CONFIG["process_pid"])
                except psutil.NoSuchProcess:
                    print(f"⚠️  未找到PID为 {MONITOR_CONFIG['process_pid']} 的进程")
            elif MONITOR_CONFIG["process_name"]:
                # 遍历所有进程，匹配进程名（支持模糊匹配）
                for p in psutil.process_iter(attrs=["pid", "name"]):
                    if p.info["name"] and MONITOR_CONFIG["process_name"].lower() in p.info["name"].lower():
                        target_p = p
                        break
                if not target_p:
                    print(f"⚠️  未找到名称包含 {MONITOR_CONFIG['process_name']} 的进程")
            else:
                return process_data  # 未配置进程信息，返回默认值

            if target_p:
                process_data = {
                    "status": target_p.status(),
                    "cpu_usage": target_p.cpu_percent(interval=0.1),
                    "mem_usage": round(target_p.memory_percent(), 2),
                    "threads": target_p.num_threads()
                }
        except Exception as e:
            print(f"⚠️  进程数据收集失败：{e}")
        return process_data

    def _collect_network(self):
        """收集端口连接状态（监控vllm端口+原项目端口）"""
        port_data = []
        try:
            connections = psutil.net_connections(kind="tcp")
            for port in MONITOR_CONFIG["monitor_ports"]:
                count = sum(
                    1 for conn in connections if conn.laddr.port == port and conn.status == psutil.CONN_ESTABLISHED)
                port_data.append({
                    "port": port,
                    "established_connections": count
                })
        except Exception as e:
            print(f"⚠️  端口数据收集失败：{e}")
            # 异常时返回端口默认连接数0
            for port in MONITOR_CONFIG["monitor_ports"]:
                port_data.append({"port": port, "established_connections": 0})
        return port_data

    def _collect_logs(self):
        """收集最新日志（修复：通过LogHandler实例访问latest_logs）"""
        if not self.log_handler:
            return []
        # 复制并清空日志，避免重复收集
        logs = self.log_handler.latest_logs.copy()
        self.log_handler.latest_logs.clear()
        return logs

    def start_log_monitor(self):
        """启动日志监听（零入侵，容错版）"""
        if not MONITOR_CONFIG["log_paths"]:
            print("⚠️  未配置日志文件路径，跳过日志监听")
            return

        valid_log_paths = []
        for log_path in MONITOR_CONFIG["log_paths"]:
            if os.path.exists(log_path):
                valid_log_paths.append(log_path)
                # 监听日志文件所在目录（监控文件修改）
                log_dir = os.path.dirname(log_path)
                self.log_observer.schedule(self.log_handler, log_dir, recursive=False)
            else:
                print(f"⚠️  日志文件不存在：{log_path}，跳过该日志监听")

        if valid_log_paths:
            self.log_observer.start()
            print(f"✅ 日志监听启动，监控文件：{valid_log_paths}")
        else:
            print("⚠️  无有效日志文件，日志监听未启动")

    def stop_log_monitor(self):
        """停止日志监听（容错版）"""
        try:
            if self.log_observer.is_alive():
                self.log_observer.stop()
                self.log_observer.join(timeout=5)
                print("✅ 日志监听已停止")
        except Exception as e:
            print(f"⚠️  停止日志监听失败：{e}")

    def collect(self):
        """统一收集所有数据源（主入口）"""
        return {
            "system": self._collect_system(),
            "gpu": self._collect_gpu(),
            "process": self._collect_process(),
            "network": self._collect_network(),
            "logs": self._collect_logs()
        }

    def __del__(self):
        """释放资源（GPU+日志监听）"""
        # 停止日志监听
        self.stop_log_monitor()
        # 释放GPU资源
        if PYNVMl_AVAILABLE and self.gpu_handles:
            try:
                pynvml.nvmlShutdown()
            except:
                pass