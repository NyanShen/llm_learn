
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
import re
import json
import time
import os

router = APIRouter(prefix="/calls", tags=["calls"])

class ThinkTagCallback(BaseCallbackHandler):
    def __init__(self):
        self.in_think = False  # 是否在<think>标签内
        self.think_buffer = ""  # 推理内容缓冲区
        self.answer_buffer = ""  # 答案缓冲区
        self.temp_buffer = ""   # 临时缓冲区（处理跨token的标签）
        
        # 标签匹配正则表达式
        self.tag_pattern = re.compile(r"(<think>|</think>)", re.IGNORECASE)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.temp_buffer += token  # 将新token添加到临时缓冲区
        
        # 切割出所有可能标签
        parts = self.tag_pattern.split(self.temp_buffer)
        self.temp_buffer = ""  # 清空临时缓冲区
        
        for part in parts:
            if not part:
                continue
                
            # 处理标签状态切换
            if part.lower() == "<think>":
                if not self.in_think:
                    self.in_think = True  # 进入<think>标签内
                continue
            elif part.lower() == "</think>":
                if self.in_think:
                    self.in_think = False
                continue
                
            # 根据状态分配内容
            if self.in_think:
                self.think_buffer += part
            else:
                self.answer_buffer += part

    def stream_data(self):
        while True:
            # 发送完整推理段落
            if self.think_buffer:
                yield f"data: {json.dumps({'type': 'reasoning', 'content': self.think_buffer})}\n\n"
                self.think_buffer = ""
                
            # 发送答案内容
            if self.answer_buffer:
                yield f"data: {json.dumps({'type': 'answer', 'content': self.answer_buffer})}\n\n"
                self.answer_buffer = ""
                
            # 结束条件（根据实际业务调整）
            if ...:  
                break
                
            time.sleep(0.1)

class ChatRequest(BaseModel):
    message: str
@router.post("/chat")
async def chat_stream(request: ChatRequest):
    print("请求开始调用.......", request.message)
    # 创建一个ChatOllama实例，用于与Ollama聊天模型进行交互
    llm = ChatOllama(
        base_url=os.environ['OLLAMA_BASE_URL'],  # 可配置为内部服务器地址
        model=os.environ['OLLAMA_LLM_MODEL_NAME']
    )
    
    async def generate():
        callback = ThinkTagCallback()
        await llm.ainvoke(request.message, callbacks=[callback])
        
        async for chunk in callback.stream_data():
            yield chunk
            
    return StreamingResponse(generate(), media_type="text/event-stream")