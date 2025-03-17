from fastapi import APIRouter, Depends
from pydantic import BaseModel
from src.services.rag_services import RagServices

router = APIRouter(prefix="/chats", tags=["chats"])

@router.get("/ask/{question}")
async def retriever_chat(question):
    return f"hello fisrt fast api your questuion is : {question}"

@router.get("/ask")
async def ask(question):
    print("请求开始调用.......")
    response = RagServices().build_qa_system(question, "mba_db")
    return {"results": response}