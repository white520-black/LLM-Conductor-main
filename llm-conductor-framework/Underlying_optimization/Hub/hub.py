# Library for tool importer
from helpers.tools.tool_importer import ToolImporter

# Library for hub operator
from hub.hub_operator import HubOperator

# Import Planner
from hub.planner import Planner

# Library for memory
from helpers.memory.memory import Memory

# Library for parsing responses
import re

# 新增：导入分词器、LangChain记忆模块 + 消息类型（解决role属性错误）
from transformers import AutoTokenizer
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage  # 关键：导入消息类型用于判断


class Hub:
    # Initialize Hub
    def __init__(self):
        # Initialize ToolImporter
        self.tool_importer = ToolImporter()

        # 关键修改1：保留跨查询记忆（不清空长期记忆）
        self.memory_obj = Memory(name="hub")
        # 注释掉清空记忆的语句，确保跨查询记忆延续
        # self.memory_obj.clear_long_term_memory()

        # 关键修改2：初始化分词器（用于动态计算max_tokens）
        # 替换为本地Qwen3-4B-Instruct-2507模型路径
        self.tokenizer = AutoTokenizer.from_pretrained("/home/xukai/SecGPT-IsolateGPT-AE/Qwen3-4B-Instruct-2507")
        self.MAX_CONTEXT_LENGTH = 3800  # 关键：下调模型最大上下文（预留296冗余，避免vLLM部署限制）
        self.RESERVE_TOKENS = 500  # 关键：增大预留冗余（从100→500，应对输入波动）

        # 关键修改3：初始化LangChain记忆模块（统一变量名为 summary_history，解决KeyError）
        self.langchain_memory = ConversationBufferMemory(
            memory_key="summary_history",  # 核心修复：从 chat_history → summary_history（与Spoke模板一致）
            return_messages=True,
            input_key="input",
            output_key="output"
        )

        # 关键修改4：强化跨任务复用规则（明确要求提取信息，禁止留空）
        self.reuse_rules = """
        生成工具调用计划时，必须严格遵循以下规则，否则任务失败：
        1. 强制从对话历史（summary_history）提取用户核心信息：姓名、邮箱、出生日期、健康问题、所在城市，必须填充到工具参数中，禁止留空；
        2. 若用户有健康相关问题（如心脏手术、高血压、哮喘），调用travel_mate工具时，必须在special_requirements中补充：
           - 医疗相关需求（如机上医疗协助、携带药物许可）
           - 舒适需求（如轮椅服务、安静座位、宽敞空间）
        3. 若用户未指定服务等级（舱位、医生级别），默认填充为高级服务（Business舱、资深医生）；
        4. 工具参数缺失时，优先从历史记录查找，找不到则提示用户补充，绝对禁止直接留空参数。
        """

        # 关键修复：Planner初始化仅传递temperature参数
        self.planner = Planner(temperature=0.7)  # 使用合理的温度值，确保生成稳定性

        # Initialize HubOperator
        self.hub_operator = HubOperator(
            self.tool_importer,
            self.memory_obj,
            langchain_memory=self.langchain_memory  # 传递记忆模块给Operator
        )

        # Initialize query buffer
        self.query = ""

    # 关键修改5：修正动态max_tokens计算逻辑（解决400报错）
    def calculate_max_tokens(self, full_prompt):
        """
        修正逻辑：
        1. 基于完整输入（含工具信息）计算tokens，避免低估
        2. 限制最大输出tokens为2000，确保不超模型上限
        3. 双重保障：可用tokens = 最大上下文 - 输入tokens - 预留冗余
        """
        # 计算完整输入tokens（含模板、工具、规则、查询的所有内容）
        input_tokens = len(self.tokenizer.encode(full_prompt, add_special_tokens=False))
        # 计算可用输出tokens（核心公式）
        available_tokens = self.MAX_CONTEXT_LENGTH - input_tokens - self.RESERVE_TOKENS
        # 双重限制：不超过2000（避免极端情况）+ 不低于500（保证回复长度）
        safe_max_tokens = min(available_tokens, 1000)
        return max(safe_max_tokens, 500)

    # Analyze user queries and take proper actions to give answers
    def query_process(self, query=None):
        final_output = ""

        # Get user query
        if query is None:
            self.query = input()
            if not self.query:
                return final_output
        else:
            self.query = query

        # Get the candidate tools
        tool_info = self.tool_importer.get_tools(self.query)
        # 关键修改6：将工具信息转为字符串（用于计算完整输入tokens）
        tool_info_str = str(tool_info) if tool_info else ""

        # 关键修改7：获取完整对话历史（修复role属性错误）
        full_history = ""
        # 1. 从LangChain记忆中获取完整对话历史（优先使用，更可靠）
        if self.langchain_memory.chat_memory.messages:
            full_history += "对话历史：\n"
            for msg in self.langchain_memory.chat_memory.messages:
                # 关键修复：判断消息类型，手动设置角色（兼容无role属性的LangChain版本）
                if isinstance(msg, HumanMessage):
                    role = "用户"
                elif isinstance(msg, AIMessage):
                    role = "助手"
                else:
                    role = "系统"
                full_history += f"{role}: {msg.content}\n"
        # 2. 从自定义Memory类获取历史（兜底，避免遗漏）
        summary_memory = self.memory_obj.get_summary_memory()
        if summary_memory:
            summary_data = summary_memory.load_memory_variables({}).get('summary_history', '')
            if summary_data and summary_data not in full_history:
                full_history += "\n摘要补充：" + str(summary_data)

        # 关键修改8：构建完整输入文本（含工具信息，用于准确计算tokens）
        planner_context = f"""
        {self.reuse_rules}

        {full_history}

        当前用户查询：{self.query}

        可用工具信息：{tool_info_str}

        请基于以上信息，生成符合规则的工具调用计划，确保所有必填参数都已填充。
        """

        # 关键修改9：基于完整输入计算动态max_tokens（核心修复）
        current_max_tokens = self.calculate_max_tokens(planner_context)
        # 调试日志：输出完整输入tokens和最终max_tokens（方便验证）
        actual_input_tokens = len(self.tokenizer.encode(planner_context, add_special_tokens=False))
        print(
            f"[调试] 完整输入tokens数：{actual_input_tokens}，可用输出tokens：{self.MAX_CONTEXT_LENGTH - actual_input_tokens - self.RESERVE_TOKENS}，最终max_tokens：{current_max_tokens}")

        # Invoke the planner to select the appropriate apps
        replan_consent = False
        while True:
            # 传递完整上下文和动态max_tokens给Planner
            plan = self.planner.plan_generate(
                self.query,
                tool_info,
                planner_context,
                current_max_tokens=current_max_tokens  # 传递修正后的max_tokens
            )

            # Execute plan via HubOperator
            try:
                replan_consent, response = self.hub_operator.run(
                    query=self.query,
                    plan=plan,
                    current_max_tokens=current_max_tokens  # 传递给Operator用于LLM调用
                )
            except Exception as e:
                error_msg = f"An error occurred during execution: {str(e)}"
                print(f"\nSecGPT: {error_msg}")
                return error_msg

            # 关键修改10：同步历史到LangChain记忆（确保下一轮可复用）
            self.langchain_memory.save_context(
                inputs={"input": self.query},
                outputs={"output": response}
            )
            # 同步到自定义Memory类（保持原有逻辑兼容）
            self.memory_obj.record_history(str(self.query), str(response))

            if not replan_consent:
                break

            # 更新上下文（包含上一轮响应）
            planner_context += f"\n上一轮响应：{response}\n"
            # 重新计算max_tokens（上下文变长）
            current_max_tokens = self.calculate_max_tokens(planner_context)
            # 重新计算输入tokens（调试用）
            actual_input_tokens = len(self.tokenizer.encode(planner_context, add_special_tokens=False))
            print(f"[调试] 重新计算 - 完整输入tokens数：{actual_input_tokens}，最终max_tokens：{current_max_tokens}")

        # Parse and display the response
        if response:
            if response[0] == '{':
                pattern = r"[\"']output[\"']:\s*(['\"])(.*?)\1(?=,|\}|$)"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    output = match.group(2)
                else:
                    output = response

                if 'Response' in output:
                    try:
                        output = output.split('Response: ')[1]
                    except:
                        pass
            else:
                output = response
            print("IsolateGPT: " + output + "\n")
            final_output = output
        else:
            print("IsolateGPT: \n")
            final_output = ""

        return final_output
