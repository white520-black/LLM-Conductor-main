# Libraries for LLMs
from langchain_openai import ChatOpenAI

# Library for importing prompt templates
from helpers.templates.prompt_templates import MyTemplates

# Library for plan parsing
from langchain_core.output_parsers import JsonOutputParser

# 新增：设置环境变量，避免tokenizers并行警告（双重保障）
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Planner:
    def __init__(self, temperature=0.0):
        # 适配本地 vLLM 部署的千问模型，配置与 vLLM 启动参数严格一致
        self.chat_llm = ChatOpenAI(
            model_name="Qwen2-72B-Instruct",  # 与 vLLM --served-model-name 完全一致
            temperature=temperature,
            model_kwargs={"seed": 0},  # 初始化时的模型参数（仅这里用 model_kwargs）
            openai_api_key="xxx",  # 与 vLLM --api-key 完全一致
            openai_api_base="http://127.0.0.1:6006/v1",  # 本地 vLLM 服务地址（含 /v1 路径）
            request_timeout=300  # 增加超时时间
        )

        # 加载提示模板
        templates = MyTemplates()
        self.template_plan = templates.template_planner

        # 初始化输出解析器（解析模型返回的 JSON 格式计划）
        self.parser = JsonOutputParser()

        # 构建基础 LLM 调用链：模板 -> 模型 -> 解析器（暂不绑定 max_tokens）
        self.base_llm_chain = self.template_plan | self.chat_llm | self.parser

        # 根据用户查询生成执行计划（支持动态传递 max_tokens）

    def plan_generate(self, query, tool_info, chat_history, current_max_tokens=None):
        # 构建调用参数（基础参数）
        invoke_params = {
            "input": query,
            "tools": tool_info,
            "chat_history": chat_history
        }

        # 关键修复：动态绑定 max_tokens（直接作为顶层参数，无需 model_kwargs）
        if current_max_tokens:
            # 临时构建带动态 max_tokens 的链
            llm_chain_with_max_tokens = self.template_plan | self.chat_llm.bind(
                max_tokens=current_max_tokens  # 直接传递 max_tokens，不嵌套
            ) | self.parser
            plan = llm_chain_with_max_tokens.invoke(invoke_params)
        else:
            # 无动态参数时使用基础链
            plan = self.base_llm_chain.invoke(invoke_params)

        return plan
