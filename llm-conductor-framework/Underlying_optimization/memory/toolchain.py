# SecGPT-IsolateGPT-AE/helpers/tools/email_tools.py
from langchain.tools import StructuredTool  # 改用旧版兼容的 StructuredTool
from helpers.tools.qq_email import QQEmailTool  # 导入核心功能类
from helpers.configs.configuration import Configs  # 导入配置类

# 初始化配置实例（从 env_variables.json 读取配置）
configs = Configs()
try:
    QQ_EMAIL_ACCOUNT = configs.env_variables["QQ_EMAIL_ACCOUNT"]
    QQ_EMAIL_AUTH_CODE = configs.env_variables["QQ_EMAIL_AUTH_CODE"]
except KeyError:
    raise ValueError("❌ 请在 data/env_variables.json 中设置 QQ_EMAIL_ACCOUNT 和 QQ_EMAIL_AUTH_CODE！")

# 初始化 QQ 邮箱工具实例
qq_email_instance = QQEmailTool(
    account=QQ_EMAIL_ACCOUNT,
    auth_code=QQ_EMAIL_AUTH_CODE
)

# -------------------------- 工具函数定义（无装饰器，纯函数）--------------------------
def get_qq_email(max_emails: int = 1) -> str:
    """
    核心功能：获取 QQ 邮箱的最近邮件，专门用于用户查询「最新邮件」「收件箱邮件」「邮件摘要」「summarize the latest email」等场景。
    参数说明：
      max_emails: 整数，可选，默认1封。表示需要获取的最近邮件数量（建议不超过10封，避免信息过多）。
    返回结果：结构化的邮件信息（包含主题、发件人、发送时间、清理后的正文），易被 AI 解析和总结。
    注意事项：仅能获取 QQ 邮箱的收件箱邮件，需确保POP3/SMTP服务已开启且授权码正确。
    """
    try:
        return qq_email_instance.get_recent_emails(max_emails=max_emails)
    except Exception as e:
        return f"❌ 调用 get_qq_email 工具失败：{str(e)}。\n" \
               "排查方向：1. 账号/授权码是否正确；2. 网络是否能访问pop.qq.com:995；3. POP3服务是否开启。"

def send_qq_email(recipient: str, subject: str, content: str) -> str:
    """
    核心功能：使用 QQ 邮箱发送邮件，适用于用户查询「发送邮件」「给某人发邮件」等场景。
    参数说明：
      recipient: 字符串，必填。收件人邮箱地址（如xxx@qq.com）。
      subject: 字符串，必填。邮件主题。
      content: 字符串，必填。邮件正文内容（纯文本）。
    返回结果：发送成功/失败的状态提示。
    注意事项：需确保SMTP服务已开启且授权码正确。
    """
    try:
        return qq_email_instance.send_email(recipient=recipient, subject=subject, content=content)
    except Exception as e:
        return f"❌ 调用 send_qq_email 工具失败：{str(e)}。\n" \
               "排查方向：1. 收件人邮箱格式是否正确；2. 账号/授权码是否正确；3. 网络是否能访问smtp.qq.com:465。"

def search_qq_email(query: str = "") -> str:
    """
    核心功能：按关键词搜索 QQ 邮箱收件箱邮件，适用于用户查询「搜索邮件」「查找包含xxx的邮件」等场景。
    参数说明：
      query: 字符串，可选，默认空。搜索关键词（匹配邮件主题、正文、发件人）。
    返回结果：匹配的邮件数量和结构化邮件信息。
    注意事项：仅搜索收件箱邮件，最多返回10封匹配结果。
    """
    try:
        return qq_email_instance.search_inbox(query=query)
    except Exception as e:
        return f"❌ 调用 search_qq_email 工具失败：{str(e)}。\n" \
               "排查方向：1. 账号/授权码是否正确；2. 网络是否能访问pop.qq.com:995；3. POP3服务是否开启。"

# -------------------------- 用 StructuredTool 封装（兼容旧版 LangChain）--------------------------
get_qq_email = StructuredTool.from_function(
    name="get_qq_email",  # 工具名称（AI 识别的关键）
    func=get_qq_email,    # 关联上面的工具函数
    description=get_qq_email.__doc__  # 直接复用函数文档作为工具描述
)

send_qq_email = StructuredTool.from_function(
    name="send_qq_email",
    func=send_qq_email,
    description=send_qq_email.__doc__
)

search_qq_email = StructuredTool.from_function(
    name="search_qq_email",
    func=search_qq_email,
    description=search_qq_email.__doc__
)