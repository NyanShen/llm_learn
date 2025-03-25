# src/main.py
import os
import sys
from dotenv import load_dotenv
# rag services
from src.services.rag_services import RagServices
# model loader
from src.services.model_loader_services import SymModelLoaderServices
from src.utils.zyyy_chunks import load_all_csv_data
# fast api
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.v1 import chats, tests, calls
from src.ws.v1 import chats as ws_chats

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
    app.include_router(calls.router, prefix="/api/v1")# 明确版本隔离

    # web socket
    app.include_router(ws_chats.router, prefix="/ws/v1")
    return app


def prepare_models():
    # SymModelLoaderServices().load_hf_embedding_model()
    # SymModelLoaderServices().load_openai_chat_model()
    # SymModelLoaderServices().load_local_chat_model()
    model_loader = SymModelLoaderServices()
    model_loader.load_ollama_embedding_model()
    model_loader.load_ollama_llm_model()
    model_loader.load_ollama_chat_model()
    print("models prepared...")

# 1.创建服务实例
app = create_app()
# 2.加载环境变量
load_dotenv()
# 3.模型加载
prepare_models()

if __name__ == "__main__":
    
    # main()
    # RagServices().build_zyyy_vector_store("data/excels", "zyyy")
    # RagServices().zyyy_adaptive_retrieval("耳鼻喉科科室电话")
    # 1.离线构建向量数据库步骤
    # RagServices().build_common_vector_store(
    #     collection_name="mba_db",
    #     overwrite=True
    # )
    # 2.在线检索向量数据库
    # response = RagServices().build_qa_system(question="简述全过程人民民主",collection_name="mba_db")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("Hello World!")
