# Libraries for LLMs
from langchain_openai import ChatOpenAI

# Library for memory
from helpers.memory.memory import Memory

# Libaries for prompt templates
from helpers.templates.prompt_templates import MyTemplates

# Libraries for agents
from langchain.agents import AgentExecutor
from spoke.output_parser import SpokeParser
from langchain_core.runnables import RunnablePassthrough

# Libraries for spoke operator
from spoke.spoke_operator import SpokeOperator

# Libraries for tools and functionalities
import json
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.tools.render import render_text_description_and_args
from helpers.tools.tool_importer import create_message_spoke_tool
from helpers.tools.tool_importer import create_function_placeholder
from helpers.tools.tool_importer import get_annotation_text

# Library for configuration
from helpers.configs.configuration import Configs

# Library for sandboxing
from helpers.sandbox.sandbox import set_mem_limit, drop_perms

# 新增：导入异常跟踪模块（打印详细报错）
import traceback


class Spoke():
    # Set up a counter to count the number of Spoke instances
    instance_count = 0

    # 关键修改1：新增 langchain_memory 和 current_max_tokens 可选参数
    def __init__(self, tool, functionalities, spec=None, temperature=0.0, flag=False, langchain_memory=None,
                 current_max_tokens=None):
        Spoke.instance_count += 1

        self.return_intermediate_steps = flag

        if tool:
            self.tools = [tool]
            self.tool_name = tool.name
        else:
            self.tools = []
            self.tool_name = ""

        self.tool_spec = spec
        self.functionalities = functionalities

        # 关键修改2：接收跨任务记忆模块（优先使用LangChain记忆）
        self.langchain_memory = langchain_memory
        # 关键修改3：接收动态max_tokens（默认2000，兼容原有逻辑）
        self.current_max_tokens = current_max_tokens or 1000

        # Initialize functionality list
        functionalities_path = Configs.functionalities_path

        with open(functionalities_path, "r") as f:
            functionality_dict = json.load(f)

        self.installed_functionalities_info = functionality_dict["installed_functionalities"]
        self.installed_functionalities = list(
            filter(lambda x: x not in self.functionalities, functionality_dict["installed_functionalities"]))

        # Create a placeholder for each functionality
        func_placeholders = create_function_placeholder(self.installed_functionalities)

        self.enabled_annotations_info = functionality_dict["enabled_annotations"]
        self.enabled_annotations = list(
            filter(lambda x: x not in self.functionalities, functionality_dict["enabled_annotations"]))

        # 关键修改4：对接本地 vLLM 千问模型，使用动态max_tokens
        self.llm = ChatOpenAI(
            model_name="Qwen2-72B-Instruct",  # 与 vLLM --served-model-name 一致
            openai_api_base="http://127.0.0.1:6006/v1",  # 本地 vLLM 服务地址
            openai_api_key="xxx",  # 与 vLLM --api-key 一致
            temperature=temperature,
            model_kwargs={"seed": 0},  # 保留种子参数确保结果可复现
            max_tokens=self.current_max_tokens  # 使用动态传递的max_tokens
        )

        # Set up memory（兼容原有逻辑，优先使用LangChain记忆）
        if self.tool_name:
            self.memory_obj = Memory(name=self.tool_name)
        else:
            self.memory_obj = Memory(name="temp_spoke")
        self.memory_obj.clear_long_term_memory()
        self.memory = self.langchain_memory or self.memory_obj.get_memory()  # 关键：优先使用跨任务记忆

        # Set up spoke operator
        self.spoke_operator = SpokeOperator(self.installed_functionalities, self.functionalities, self.tool_spec)

        # Set up prompt templates
        self.templates = MyTemplates()
        if self.tool_name in self.enabled_annotations_info:
            self.prompt = self.templates.annotation_spoke_prompt

            self.prompt = self.prompt.partial(
                tools=get_annotation_text([self.tool_name]) + render_text_description_and_args(list(func_placeholders)),
            )

            tool_functionality_list = func_placeholders

        else:
            self.prompt = self.templates.spoke_prompt

            missing_vars = {"tools", "tool_names", "agent_scratchpad"}.difference(
                self.prompt.input_variables
            )
            if missing_vars:
                raise ValueError(f"Prompt missing required variables: {missing_vars}")

            tool_functionality_list = self.tools + func_placeholders
            self.prompt = self.prompt.partial(
                tools=render_text_description_and_args(list(tool_functionality_list)),
                tool_names=", ".join([t.name for t in tool_functionality_list]),
            )

        self.llm_with_stop = self.llm.bind(stop=["Observation"])
        tool_functionality_list.append(create_message_spoke_tool())

        self.agent = (
                RunnablePassthrough.assign(
                    agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
                )
                | self.prompt
                | self.llm_with_stop
                | SpokeParser(functionality_list=self.installed_functionalities, spoke_operator=self.spoke_operator)
        )

        self.agent_chain = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=tool_functionality_list,
            verbose=Configs.debug_mode.value,
            memory=self.memory,  # 关键：使用跨任务记忆
            handle_parsing_errors=True,
            return_intermediate_steps=self.return_intermediate_steps
        )

    # 关键修改5：execute方法修复（动态max_tokens格式 + 详细报错日志）
    def execute(self, request, entities, current_max_tokens=None):
        # 动态更新max_tokens（优先级：当前调用参数 > 实例参数）
        # 关键修复：直接传递max_tokens，不使用model_kwargs（兼容vLLM API + LangChain参数要求）
        if current_max_tokens:
            self.llm = self.llm.bind(max_tokens=current_max_tokens)

        try:
            results = self.agent_chain.invoke({'input': request, 'entities': entities},
                                              return_only_outputs=not self.return_intermediate_steps)
        except Exception as e:
            # 关键修复：打印详细报错信息（包括堆栈跟踪），方便定位问题
            error_detail = traceback.format_exc()
            print(f"[Spoke执行错误] 工具：{self.tool_name}，请求：{request}，错误详情：\n{error_detail}")
            results = {"output": f"An error occurred during spoke execution: {str(e)}"}
        finally:
            return results

    # 关键修改6：run_process新增 langchain_memory 和 current_max_tokens 参数
    def run_process(self, child_sock, request, spoke_id, entities, langchain_memory=None, current_max_tokens=None):
        # Set seccomp and setrlimit
        set_mem_limit()
        drop_perms()

        # 更新实例的记忆模块和max_tokens（接收从HubOperator传递的参数）
        if langchain_memory:
            self.langchain_memory = langchain_memory
            self.memory = self.langchain_memory  # 同步更新Agent记忆
        if current_max_tokens:
            self.current_max_tokens = current_max_tokens

        self.spoke_operator.spoke_id = spoke_id
        self.spoke_operator.child_sock = child_sock
        request_formatted, request = self.spoke_operator.parse_request(request)
        if request_formatted:
            # 关键：传递动态max_tokens给execute方法
            results = self.execute(request, entities, current_max_tokens=self.current_max_tokens)
        else:
            results = {"output": "Invalid request format. Please provide a valid json blob."}
        self.spoke_operator.return_response(results, request_formatted, self.return_intermediate_steps)

    @classmethod
    def get_instance_count(cls):
        return cls.instance_count