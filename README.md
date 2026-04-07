# LLM-Conductor-main
An adaptive LLM application framework featuring closed-loop decision-making, cross-session policy memory, and dynamic resource scheduling for resource-constrained environments.
LLM-Conductor is a three-layer collaborative architecture designed to address critical deployment challenges of Large Language Models (LLMs) in industrial environments, including weak autonomous decision-making, low resource utilization efficiency, and high deployment invasiveness. The framework integrates a monitoring-feedback closed loop, policy memory reuse, and resource-constrained scheduling to achieve end-to-end automatic control for LLM applications in dynamic environments.
Built upon a "Policy Optimization - Foundational Support - Theoretical Guarantee" architecture, LLM-Conductor enables secure, efficient, and adaptive deployment of LLM agents across cloud servers, edge nodes, and lightweight IoT devices.
CORE FEATURES
C1 - Monitoring-Feedback Autonomous Loop
Real-time system state perception using Isolation Forest and DBSCAN for anomaly detection, combined with LLM deep reasoning for adaptive policy generation. Implements zero-intrusion data collection via psutil and pynvml with <1ms inference latency.
C2 - Policy Memory Graph & Cross-Task Transfer
Redis-based structured graph storage for task types, scenario features, and tool invocation sequences. Enables efficient cross-task experience reuse through TF-IDF similarity matching (threshold >0.7) with <50ms transfer latency.
C3 - MDP-Based Multi-Tool RL Optimization
Models tool invocation sequences as Markov Decision Processes using Proximal Policy Optimization (PPO). Achieves global optimization of toolchain combinations through offline pre-training and online fine-tuning, balancing task efficiency and resource consumption.
C4 - Cross-Session Memory Persistence
Classified storage of conversation history, entity information, and core summaries using Redis. Implements hybrid RDB+AOF persistence with LRU caching, maintaining read latency <100ms and average entry size <5KB.
C5 - Context-Aware Dynamic Resource Allocation
Precise token counting using Qwen tokenizer with dynamic output quota allocation. Prevents token overflow through adaptive calculation: N_output = min(N_model_max - N_input - N_safety, N_task_limit), maintaining >85% token utilization.
C6 - Elastic Model Scheduling for Resource-Constrained Environments
Three-tier adaptive scheduling (Full/Simplified/Lightweight toolchains) based on real-time resource monitoring. Maintains >95% task completion rate under constrained resources (1 CPU core @ 0.5GHz, 512MB RAM).
C7 - Zero-Intrusion Deployment Framework
Non-invasive data acquisition through system call layer isolation (psutil/nvidia-ml-py3), filesystem layer isolation (watchdog inotify), and process space isolation. Guarantees <5% overhead and zero write/inject operations on host services.
ARCHITECTURE
Strategy Optimization Layer
Monitoring-Feedback Loop: ML anomaly detection + LLM reasoning
Policy Memory Graph: Redis-based structured experience storage
MDP-Based Optimization: PPO reinforcement learning for toolchain optimization
Foundational Support Layer
Redis-Based Persistent Storage: Cross-session memory management
Qwen Tokenizer Resource Allocation: Dynamic token budgeting
Low-Compute Elastic Scheduling: Tiered resource adaptation
Theoretical Guarantee Layer
Formal Observability Conditions: Mathematical guarantees for zero-intrusion
Agentless Data Acquisition: Non-invasive monitoring protocols
SYSTEM REQUIREMENTS
Hardware
CPU: Intel Xeon Gold 5317 or equivalent (32 cores recommended, supports down to 1 core @ 0.5GHz)
Memory: 128GB recommended (minimum 512MB for constrained deployment)
GPU: NVIDIA A40 48GB x2 (optional, supports CPU-only mode)
Storage: 2TB SSD
Software
OS: Ubuntu 22.04 LTS (kernel 5.15.0-78-generic)
Python: 3.9.16
Redis: 7.0+ (with RedisGraph 2.0+)
CUDA: 13.0 (optional)
DEPENDENCIES
Core Framework:
langchain==0.1.10
langchain-openai
langchain-community
Machine Learning & RL:
scikit-learn==1.3.2
stable-baselines3==2.0.0
gymnasium
System Monitoring:
psutil==5.9.8
pynvml==11.5.0
watchdog==3.0.0
Model Serving:
vllm==0.4.0
transformers
accelerate
Data & Storage:
redis
numpy
pandas
faiss-cpu
Utilities:
jsonschema
python-dotenv
INSTALLATION
Clone the repository:
git clone https://github.com/your-repo/llm-conductor.git
cd llm-conductor
Install dependencies:
pip install -r requirements.txt
Configure Redis:
Install Redis 7.0+ and RedisGraph 2.0+
Update connection string in helpers/configs/configuration.py
Configure LLM endpoint:
Deploy Qwen2-72B-Instruct via vLLM:
vllm serve Qwen2-72B-Instruct --tensor-parallel-size 2 --api-key xxx --port 6006
Update API configuration in monitor/config.py and helpers/configs/
(Optional) Configure resource constraints:
Set cgroup limits for zero-intrusion monitoring:
CPU quota: 10%
Memory limit: 512MB
USAGE
Quick Start:
from hub.hub import Hub
Initialize framework
hub = Hub()
Process query with closed-loop monitoring
response = hub.query_process("Book a medical appointment and find nearby pharmacies")
Running Experiments:
Security Validation (2,004 test cases):
python experiments/security_validation.py --dataset isolategpt_sec
Functional Completeness:
python experiments/functional_validation.py --benchmark langchain
Resource-Constrained Adaptation (6 gradients):
python experiments/resource_adaptation.py --gradient 4
Monitor Mode:
python -m llm_sentry.main start --duration 3600
Configuration:
Key configuration files:
helpers/configs/configuration.py: Redis, LLM endpoints, tool specifications
monitor/config.py: Monitoring parameters, sampling frequency (1Hz), thresholds
data/env_variables.json: API keys and credentials
EXPERIMENTAL RESULTS
Security Performance:
Risk Incidence Rate: Reduced from 70.6% to 1.3% (97.8% improvement)
Anomaly Detection Rate: 96.7% (exceeds 90% industry standard)
Defense Response Time: 78.9ms (within 100ms real-time requirement)
False Positive Rate: 1.8% (below 5% threshold)
Functional Completeness:
Multi-Application Collaboration Completion: 100% (vs. 65% baseline)
Output Format Error Rate: 4.8% (vs. 28.6% baseline)
Cross-Application Data Mismatch: Eliminated (0%)
Resource Efficiency:
Token Utilization: >85% across all scenarios (vs. 25.9% baseline)
CPU Utilization: Stabilized at 60% (threshold 80%)
GPU Load Balancing: Dual-GPU synchronized at 60-80%
Non-Invasive Overhead: <3% performance penalty
Resource-Constrained Adaptation:
Gradient 1 (4-core/8GB/GPU): 100% completion
Gradient 2 (2-core/4GB/GPU): 100% completion
Gradient 3 (1-core/2GB/CPU): 100% completion
Gradient 4 (1-core/512MB/CPU): 95% completion
Gradient 5 (1-core/256MB/CPU): 74% completion
Gradient 6 (1-core/128MB/CPU): 0% (initialization failure)
Operational Boundary: 1 CPU core @ 0.5GHz, 512MB RAM
