# src/main.py
import os
import sys
from dotenv import load_dotenv
# rag services
from src.services.rag_services import RagServices
# model loader
from src.core.model_loader import ModelLoader
# fast api
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.v1 import chats, tests

def create_app():
    app = FastAPI(title="SymLLM")
    # CORS跨域访问问题
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # register router
    app.include_router(tests.router, prefix="/api/v1")# 明确版本隔离
    app.include_router(chats.router, prefix="/api/v1")# 明确版本隔离
    return app


def prepare_models():
    ModelLoader().load_hf_embedding_model()
    ModelLoader().load_openai_chat_model()
    print("models prepare")

# 1.创建服务实例
app = create_app()
# 2.加载环境变量
load_dotenv()
# 3.模型加载
prepare_models()

if __name__ == "__main__":
    # RagServices().build_qa_system("简述高质量发展")
    # build_vector_store()
    # main()
    print("Hello World!")
