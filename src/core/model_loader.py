
import os
from langchain_openai import ChatOpenAI 
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer # 加载和使用Embedding模型
from dotenv import load_dotenv

class ModelLoader:
    def __init__(self):
        """初始化"""
        load_dotenv()
    
    @staticmethod
    def load_openai_chat_model():
        """加载chatllm模型"""
        return ChatOpenAI(
            base_url=os.getenv('MODEL_BASE_URL'),
            api_key=os.environ['MODEL_API_KEY'],
            model=os.environ['MODEL_NAME'],
            temperature=os.getenv('MODEL_TEMPERATURE', 0.2),
            max_tokens=os.getenv('MODEL_MAX_TOKENS', 20),
        )
    
    @staticmethod
    def load_hf_embedding_model():
        """
        加载本地embedding模型
        :return: 返回加载的embedding模型
        chroma
        """
        model_name=os.environ['EMBEDDING_MODEL_PATH']
        model_kwargs={
            "device": "cpu"
        }
        encode_kwargs={
            "normalize_embeddings": True
        }
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
    @staticmethod
    def load_st_embedding_model():
        """
        加载bge-small-zh-v1.5模型
        :return: 返回加载的bge-small-zh-v1.5模型
        用法：embedding_model.encode 配合faiss使用
        """
        print(f"加载Embedding模型中")
        # SentenceTransformer读取绝对路径下的bge-small-zh-v1.5模型，非下载
        # embedding_model = SentenceTransformer(os.path.abspath(os.environ['EMBEDDING_SMALL_MODEL_PATH']))
        model_path = os.environ['EMBEDDING_SMALL_MODEL_PATH']
        embedding_model = SentenceTransformer(model_name_or_path=model_path)
        print(f"bge-small-zh-v1.5模型最大输入长度: {embedding_model.max_seq_length}") 
        return embedding_model