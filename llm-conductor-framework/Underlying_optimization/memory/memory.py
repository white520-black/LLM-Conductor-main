# 导入必要的库
from transformers import AutoTokenizer
import warnings
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryBufferMemory,
    ConversationEntityMemory,
    CombinedMemory
)
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage  # 补充消息类型导入
from typing import List  # 新增类型注解支持
from helpers.configs.configuration import Configs

# 步骤1：加载本地千问模型的分词器（使用绝对路径）
try:
    # 加载本地Qwen3-4B-Instruct-2507模型的分词器
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        "/home/xukai/SecGPT-IsolateGPT-AE/Qwen3-4B-Instruct-2507",  # 本地模型绝对路径
        trust_remote_code=True  # 千问分词器需要自定义代码支持
    )
except Exception as e:
    warnings.warn(f"千问分词器加载失败: {str(e)}。将使用备用分词逻辑（可能导致token计数不准确）。")
    qwen_tokenizer = None


# 步骤2：自定义token计数函数（基于千问分词器）
def qwen_get_num_tokens(text: str) -> int:
    """计算单段文本的token数量（使用千问分词器）"""
    if qwen_tokenizer is None:
        return len(text.split())  # 备用逻辑：简单按空格分词
    # 不添加特殊token（如<bos>、<eos>），避免重复计数
    return len(qwen_tokenizer.encode(text, add_special_tokens=False))


def qwen_get_num_tokens_from_messages(messages) -> int:
    """计算对话消息列表的总token数量（使用千问分词器）"""
    if qwen_tokenizer is None:
        return sum(len(str(msg).split()) for msg in messages)  # 备用逻辑
    # 将消息转换为"role: content"格式后编码
    text = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
    return len(qwen_tokenizer.encode(text, add_special_tokens=False))


# 步骤3：重写SummaryBufferMemory，适配千问分词器（核心修复prune方法）
class QwenConversationSummaryBufferMemory(ConversationSummaryBufferMemory):
    """自定义对话摘要内存类，使用千问分词器计算token，修复prune方法"""

    def __init__(self, **kwargs):
        """初始化，确保父类参数正确传递（尤其是llm）"""
        super().__init__(**kwargs)

    def _get_num_tokens(self, text: str) -> int:
        return qwen_get_num_tokens(text)

    def get_num_tokens_from_messages(self, messages) -> int:
        return qwen_get_num_tokens_from_messages(messages)

    def _create_summary(self, messages: List) -> str:
        """生成对话历史的摘要（千问模型适配版）"""
        # 1. 格式化对话历史为"Human: ...\nAI: ..."格式
        formatted_dialog = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_dialog.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                formatted_dialog.append(f"助手: {msg.content}")
        dialog_text = "\n".join(formatted_dialog)

        # 2. 构造千问模型的摘要提示词（适配中文表达习惯）
        summary_prompt = (
            "请总结以下对话的核心内容，保留关键信息，用简洁的中文概括：\n"
            "对话历史：\n"
            f"{dialog_text}\n"
            "总结："
        )

        # 3. 调用千问模型生成摘要（使用llm.predict确保同步返回）
        try:
            summary = self.llm.predict(summary_prompt)
            return summary.strip()  # 去除首尾空格
        except Exception as e:
            warnings.warn(f"千问模型生成摘要失败: {str(e)}，使用简化摘要替代")
            return f"对话摘要：包含{len(messages)}条消息，涉及用户与助手的交互内容。"

    def prune(self) -> None:
        """重写修剪方法，使用自定义的token计数和摘要生成逻辑"""
        # 获取当前对话历史
        buffer = self.chat_memory.messages
        if not buffer:  # 空历史无需修剪
            return

        # 计算当前对话的token长度
        curr_buffer_length = self.get_num_tokens_from_messages(buffer)

        # 如果超过最大限制，生成摘要并修剪
        if curr_buffer_length > self.max_token_limit:
            summary = self._create_summary(buffer)  # 调用新增的摘要生成方法
            self.chat_memory.clear()
            self.chat_memory.add_message(HumanMessage(content=summary))
        return


class Memory:
    """内存管理类，整合多种内存类型并适配千问模型"""

    def __init__(self, name):
        # 数据库配置
        db_url = Configs.db_url

        # 初始化对接本地vLLM千问模型的LLM（用于生成对话摘要和实体提取）
        self.llm = ChatOpenAI(
            model_name="Qwen2-72B-Instruct",  # 与vLLM的--served-model-name一致
            openai_api_base="http://127.0.0.1:6006/v1",  # 本地vLLM服务地址
            openai_api_key="xxx",  # 与vLLM的--api-key一致
            temperature=0.0,  # 摘要和实体提取使用确定性模式
            model_kwargs={"seed": 0},
            max_tokens=100  # 避免超过模型最大长度限制
        )

        # 初始化Redis存储的消息历史（分别存储完整对话、摘要、实体）
        self.message_history = RedisChatMessageHistory(
            url=db_url,
            ttl=600,  # 消息过期时间（10分钟）
            session_id=name
        )
        self.summary_history = RedisChatMessageHistory(
            url=db_url,
            ttl=600,
            session_id=f"{name}_summary"
        )
        self.entity_history = RedisChatMessageHistory(
            url=db_url,
            ttl=600,
            session_id=f"{name}_entity"
        )

        # 1. 完整对话内存（不修剪，存储所有历史）
        self.conv_memory = ConversationBufferMemory(
            chat_memory=self.message_history,
            memory_key="buffer_history",
            input_key="input",
            output_key="output"
        )

        # 2. 摘要对话内存（使用千问分词器计算token，超过限制时修剪）
        self.summary_memory = QwenConversationSummaryBufferMemory(
            llm=self.llm,  # 传入千问模型用于生成摘要
            max_token_limit=300,  # 基于千问token的最大限制（可根据需求调整）
            memory_key="summary_history",
            input_key="input",
            output_key="output",
            chat_memory=self.summary_history
        )

        # 3. 实体内存（存储对话中的实体信息）
        self.entity_memory = ConversationEntityMemory(
            llm=self.llm,  # 使用千问模型提取实体
            chat_memory=self.entity_history,
            chat_history_key="entity_history",
            input_key="input",
            output_key="output",
            get_num_tokens=qwen_get_num_tokens  # 使用千问token计数
        )

        # 合并所有内存类型
        self.memory = CombinedMemory(
            memories=[self.conv_memory, self.summary_memory, self.entity_memory]
        )

    def get_memory(self):
        """获取合并后的内存对象"""
        return self.memory

    def get_entity_memory(self):
        """获取实体内存对象"""
        return self.entity_memory

    def get_summary_memory(self):
        """获取摘要内存对象"""
        return self.summary_memory

    def get_long_term_full_memory(self):
        """获取长期存储的完整对话历史"""
        return self.message_history.messages

    def get_long_term_summary_memory(self):
        """获取长期存储的对话摘要历史"""
        return self.summary_history.messages

    def get_long_term_entity_memory(self):
        """获取长期存储的实体历史"""
        return self.entity_history.messages

    def clear_long_term_memory(self):
        """清空所有长期存储的内存"""
        self.message_history.clear()
        self.summary_history.clear()
        self.entity_history.clear()

    def retrieve_entities(self, data):
        """从实体内存中检索与输入相关的实体"""
        _input = {"input": data}
        entity_dict = self.entity_memory.load_memory_variables(_input)
        return str(entity_dict.get("entities", {}))

    def record_history(self, inputs, outputs):
        """记录对话历史到内存"""
        inputs_dict = {"input": inputs}
        outputs_dict = {"output": outputs}
        self.memory.save_context(inputs_dict, outputs_dict)