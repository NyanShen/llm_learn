import asyncio
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from src.services.rag_services import RagServices
from src.services.mba_rag_chain import SymMBARAGChain

router = APIRouter(prefix="/chats", tags=["chats"])

@router.get("/ask/{question}")
async def retriever_chat(question):
    return f"hello fisrt fast api your questuion is : {question}"

@router.get("/ask")
async def ask(question):
    print("请求开始调用.......")
    response = RagServices().build_qa_system(question, "mba_db")
    return {"results": response}

class ChatRequest(BaseModel):
    message: str
@router.post("/streamtest")
async def chat_stream(request: ChatRequest):
    print("请求开始调用.......", request.message)

    async def event_stream():
        try:
            # 模拟流式输出
            # response_text = f"这是一个流式响应示例，实际需要接入LangChain,您的输入是：{request.message}"
            # for char in response_text:
            #     # sse流式输出格式要求“data:{content}\n\n”
            #     yield f"data: {char}\n\n"
            #     await asyncio.sleep(0.05)
            
            # 初始化你的LangChain组件
            chain = SymMBARAGChain().get_chain()  # 这里替换为实际初始化代码
            # 实际使用时应该从LangChain获取流式输出
            async for chunk in chain.astream(request.message):
                yield f"data: {chunk}\n\n"
            
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"}
    )
