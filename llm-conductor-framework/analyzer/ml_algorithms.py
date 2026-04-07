import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class MLAnalyzer:
    def __init__(self):
        # 1. 隔离森林（Isolation Forest）：资源异常检测（CPU/GPU/内存）
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=0.05,  # 异常比例阈值
            random_state=42
        )
        self.scaler = StandardScaler()
        self.resource_model_trained = False

        # 2. DBSCAN：日志异常检测（文本聚类）
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.log_model_trained = False

    def train_resource_model(self, baseline_data):
        """用基线数据训练资源异常检测模型（适配无GPU）"""
        # 提取特征：CPU使用率、内存使用率（无GPU时忽略GPU特征）
        features = []
        for data in baseline_data:
            cpu = data["system"]["cpu_usage"]
            mem = data["system"]["mem_usage"]
            # 判断是否有GPU数据
            if data["gpu"]:
                gpu_avg = np.mean([g["gpu_usage"] for g in data["gpu"]])
                features.append([cpu, mem, gpu_avg])
            else:
                features.append([cpu, mem])  # 无GPU时只保留CPU和内存

        if len(features) < 10:
            raise ValueError("基线数据不足，无法训练模型")

        features_scaled = self.scaler.fit_transform(features)
        self.isolation_forest.fit(features_scaled)
        self.resource_model_trained = True

    def detect_resource_anomaly(self, data):
        """检测资源异常（适配无GPU）"""
        if not self.resource_model_trained:
            return False

        cpu = data["system"]["cpu_usage"]
        mem = data["system"]["mem_usage"]
        # 判断是否有GPU数据
        if data["gpu"]:
            gpu_avg = np.mean([g["gpu_usage"] for g in data["gpu"]])
            feature = self.scaler.transform([[cpu, mem, gpu_avg]])
        else:
            feature = self.scaler.transform([[cpu, mem]])

        return self.isolation_forest.predict(feature)[0] == -1  # -1表示异常

    def train_log_model(self, baseline_logs):
        """用基线日志训练日志异常检测模型（兼容无日志场景）"""
        # 提取所有非空日志
        log_texts = [log for logs in baseline_logs for log in logs if log.strip()]

        # 日志不足时，跳过训练（不抛出错误）
        if len(log_texts) < 20:
            print("⚠️  基线日志不足（需至少20条有效日志），跳过日志异常检测模型训练")
            self.log_model_trained = False
            return

        # 正常训练日志模型
        tfidf_matrix = self.tfidf.fit_transform(log_texts)
        self.dbscan.fit(tfidf_matrix)
        self.log_model_trained = True
        print("✅ 日志异常检测模型训练完成")

    def detect_log_anomaly(self, logs):
        """检测日志异常（适配无日志模型场景）"""
        if not self.log_model_trained or not logs:
            return []

        # 编码当前日志并检测
        tfidf_matrix = self.tfidf.transform([log.strip() for log in logs if log.strip()])
        labels = self.dbscan.fit_predict(tfidf_matrix)
        abnormal_logs = [logs[i] for i, label in enumerate(labels) if label == -1]
        return abnormal_logs