# Libraries for processing json
import sys
from jsonschema import validate
import ast
import json

# Libraries for messages
from helpers.isc.message import Message


class SpokeOperator:
    def __init__(self, functionality_list, functionalities, tool_spec):
        self.functionality_list = functionality_list
        self.tool_spec = tool_spec if tool_spec else {}  # 容错：避免 tool_spec 为 None
        self.functionalities = functionalities
        self.spoke_id = None
        self.child_sock = None
        self.request_functionality = None  # 初始化：避免后续访问空值

    def parse_request(self, request):
        is_formatted = True
        try:
            if request.startswith('{'):
                # 优先用 json.loads 解析（更规范，支持标准JSON）
                request_dict = json.loads(request)
                self.request_functionality = request_dict.get('functionality_request')  # 安全获取，避免KeyError
                request_body = request_dict.get('request_body', {})

                if self.tool_spec and self.request_functionality:
                    # 关键：添加工具spec层级检查，避免KeyError
                    if (self.request_functionality in self.tool_spec.get('properties', {}) and
                            'properties' in self.tool_spec['properties'][self.request_functionality] and
                            'request' in self.tool_spec['properties'][self.request_functionality]['properties']):
                        request_schema = self.tool_spec['properties'][self.request_functionality]['properties'][
                            'request']
                        is_formatted = self.check_format(request_schema, request_body)
                        if is_formatted:
                            data = ', '.join(f"{key}={repr(value)}" for key, value in request_body.items())
                            request = f"{self.request_functionality}({data})"
                    else:
                        is_formatted = False  # spec格式不匹配，标记为未格式化
            return is_formatted, request

        except json.JSONDecodeError:
            # JSON解析失败，尝试用 ast.literal_eval 兼容非标准JSON
            try:
                request_dict = ast.literal_eval(request)
                self.request_functionality = request_dict.get('functionality_request')
                request_body = request_dict.get('request_body', {})

                if self.tool_spec and self.request_functionality:
                    if (self.request_functionality in self.tool_spec.get('properties', {}) and
                            'properties' in self.tool_spec['properties'][self.request_functionality] and
                            'request' in self.tool_spec['properties'][self.request_functionality]['properties']):
                        request_schema = self.tool_spec['properties'][self.request_functionality]['properties'][
                            'request']
                        is_formatted = self.check_format(request_schema, request_body)
                        if is_formatted:
                            data = ', '.join(f"{key}={repr(value)}" for key, value in request_body.items())
                            request = f"{self.request_functionality}({data})"
                    else:
                        is_formatted = False
                return is_formatted, request
            except Exception as e:
                # 所有解析失败，重置 request_functionality，标记未格式化
                self.request_functionality = None
                is_formatted = False
                return is_formatted, request
        except Exception as e:
            # 其他异常，重置 request_functionality
            self.request_functionality = None
            is_formatted = False
            return is_formatted, request

    # Format and send the probe message to the hub
    def probe_functionality(self, functionality: str):
        # 检查功能是否在列表中
        if not functionality or functionality not in self.functionality_list:
            return None, None, None

        # 格式化探测消息
        probe_message = Message().function_probe_request(self.spoke_id, functionality)
        self.child_sock.send(probe_message)
        response = self.child_sock.recv()

        if response.get('message_type') == 'function_probe_response':
            functionality_offered = response.get('functionality_offered', {})
            # 安全获取 request_schema 和 response_schema，避免KeyError
            if (functionality in functionality_offered.get('properties', {}) and
                    'properties' in functionality_offered['properties'][functionality]):
                request_schema = functionality_offered['properties'][functionality]['properties'].get('request')
                response_schema = functionality_offered['properties'][functionality]['properties'].get('response')
            else:
                request_schema = None
                response_schema = None
        else:
            request_schema = None
            response_schema = None

        return response.get('message_type'), request_schema, response_schema

    # Format and send the app request message to the hub
    def make_request(self, functionality: str, request: dict):
        if not functionality or not isinstance(request, dict):
            return 'invalid_request', "Invalid functionality or request format"

        app_request_message = Message().app_request(self.spoke_id, functionality, request)
        self.child_sock.send(app_request_message)
        response = self.child_sock.recv()

        return response.get('message_type'), response.get('response', {})

    def check_format(self, schema, instance_dict):
        """安全验证实例是否符合schema，处理字典/字符串格式转换"""
        try:
            # 如果实例是字符串，尝试解析为字典（工具可能返回JSON字符串）
            if isinstance(instance_dict, str):
                try:
                    instance_dict = json.loads(instance_dict)
                except:
                    # 解析失败，若schema要求是string则通过，否则失败
                    return schema.get('type') == 'string'

            # 验证实例是否符合schema
            validate(instance=instance_dict, schema=schema)
            return True
        except Exception as e:
            print(f"[格式验证失败] 预期schema：{schema}，实际实例：{instance_dict}，错误：{str(e)}")
            return False

    def return_response(self, results, request_formatted, return_intermediate_steps):
        """安全返回响应，避免KeyError，适配结构化响应格式"""
        # 默认响应：避免返回空值
        default_response = "No valid response generated."
        try:
            if not return_intermediate_steps:
                validating_output = results.get('output', default_response)

                # 只有 tool_spec 存在且请求格式化成功时，才进行格式验证
                if self.tool_spec and request_formatted:
                    response_schema = None
                    # 优先使用 request_functionality（精准匹配）
                    if self.request_functionality:
                        if (self.request_functionality in self.tool_spec.get('properties', {}) and
                                'properties' in self.tool_spec['properties'][self.request_functionality]):
                            response_schema = self.tool_spec['properties'][self.request_functionality][
                                'properties'].get('response')
                    # 若 request_functionality 为空，遍历所有功能查找匹配的schema
                    else:
                        for func in self.functionalities:
                            if (func in self.tool_spec.get('properties', {}) and
                                    'properties' in self.tool_spec['properties'][func]):
                                response_schema = self.tool_spec['properties'][func]['properties'].get('response')
                                break

                    # 若找到 response_schema，验证格式；未找到则直接返回结果
                    if response_schema:
                        # 处理工具返回的结构化数据（可能是字典或JSON字符串）
                        if isinstance(validating_output, dict):
                            is_formatted = self.check_format(response_schema, validating_output)
                        elif isinstance(validating_output, str):
                            # 尝试解析字符串为字典（工具可能返回JSON字符串）
                            try:
                                validating_output_dict = json.loads(validating_output)
                                is_formatted = self.check_format(response_schema, validating_output_dict)
                            except:
                                is_formatted = self.check_format(response_schema, validating_output)
                        else:
                            is_formatted = False

                        if is_formatted:
                            # 格式验证通过，返回结构化响应
                            response = Message().final_response(self.spoke_id, validating_output)
                            self.child_sock.send(response)
                            return
                        else:
                            # 格式不匹配，但返回实际结果（避免用户看不到有效信息）
                            results = f"Response format warning: Expected format {response_schema}. Actual response: {validating_output}"
                    else:
                        # 无 response_schema，直接返回结果
                        results = validating_output

            # 处理中间步骤或格式验证失败的情况
            final_response = results if isinstance(results, str) else str(results)
            response = Message().final_response(self.spoke_id, final_response)
            self.child_sock.send(response)

        except Exception as e:
            # 捕获所有异常，避免进程崩溃，返回明确错误信息
            error_msg = f"Error returning response: {str(e)}"
            print(f"[返回响应异常] {error_msg}")
            response = Message().final_response(self.spoke_id, error_msg)
            self.child_sock.send(response)