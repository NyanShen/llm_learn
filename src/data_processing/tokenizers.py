# 结构化输出的token
import json
from fastapi import WebSocket
from langchain.callbacks.base import BaseCallbackHandler
from pydantic import BaseModel

class StructuredStreamingCallback(BaseCallbackHandler):
    def __init__(self, websocket: WebSocket):
        super().__init__()
        self.in_thinking = False  # 标记是否处于<think>推理过程中
        self.buffer = ""          # 缓存可能被分割的标签（如"<think"和">"分两次生成）
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 1. 检测标签并更新状态
        self._detect_tag(token)
        
        # 2. 跳过标签本身的输出（只保留内容）
        if token.strip() in ["<think>", "</think>"]:
            return
        
        # 3. 根据状态构建结构化输出
        output_type = "reasoning" if self.in_thinking else "answer"
        structured_data = { "type": output_type,  "content": token }
        # 4. 实时输出JSON格式（实际场景可改为WebSocket推送或保存到队列）
        # print(json.dumps(structured_data), flush=True)
        # 通过 WebSocket 异步发送数据
        await self.websocket.send_text(json.dumps(structured_data))

    async def on_llm_end(self, response, **kwargs):
        structured_data = { "type": "done",  "content": "[done]" }
        await self.websocket.send_text(json.dumps(structured_data))
        await self.websocket.close()  # 结束时关闭连接

    def _detect_tag(self, token: str) -> None:
        """检测<think>和</think>标签，更新状态"""
        self.buffer += token
        
        # 检测是否开始推理（可能因流式生成被拆分成多个token）
        if "<think>" in self.buffer:
            self.in_thinking = True
            self.buffer = self.buffer.replace("<think>", "")  # 清理已处理的标签
        
        # 检测是否结束推理
        if "</think>" in self.buffer:
            self.in_thinking = False
            self.buffer = self.buffer.replace("</think>", "")