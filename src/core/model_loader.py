
import os
import functools
import threading
from langchain_openai import ChatOpenAI 
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from sentence_transformers import SentenceTransformer # 加载和使用Embedding模型
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv

def cached_model(model_key):
    '''
    缓存模型装饰器
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper():
            with ModelLoader._lock:
                if model_key not in ModelLoader._loaded_models:
                    ModelLoader._loaded_models[model_key] = func()
                return ModelLoader._loaded_models[model_key]
        return wrapper
    return decorator


class ModelLoader:
    _loaded_models = {}  # 缓存已加载的模型实例
    _lock = threading.Lock()  # 线程锁

    def __init__(self):
        """初始化"""
        load_dotenv()

    @staticmethod
    @cached_model("openai_chat_model")
    def load_openai_chat_model():
        """加载chatllm模型"""
        print("加载chatllm模型:", os.getenv('MODEL_BASE_URL'))
        chat_model = ChatOpenAI(
            base_url=os.getenv('MODEL_BASE_URL'),
            api_key=os.environ['MODEL_API_KEY'],
            model=os.environ['MODEL_NAME'],
            temperature=os.getenv('MODEL_TEMPERATURE', 0.2),
            max_tokens=os.getenv('MODEL_MAX_TOKENS', 20),
        )
        print("加载chatllm模型完成")
        return chat_model
    @staticmethod
    @cached_model("local_chat_model")
    def load_local_chat_model():
        """加载本地chatllm模型"""
        print("加载本地chatllm模型:")
        # 加载分词器和模型
        model_path = os.environ['MODEL_LOCAL_PATH']
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # 创建推理管道
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=200,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1
        )

        # 创建LangChain的LLM实例
        local_chat_model = HuggingFacePipeline(pipeline=pipe)

        print("加载本地chatllm模型完成", local_chat_model)
        return local_chat_model
    
    @staticmethod
    @cached_model("hf_embedding_model")
    def load_hf_embedding_model():
        """
        加载本地embedding模型
        :return: 返回加载的embedding模型
        chroma
        """
        model_name=os.environ['EMBEDDING_SMALL_MODEL_PATH']
        model_kwargs={
            "device": "cpu"
        }
        encode_kwargs={
            "normalize_embeddings": True
        }
        print(f"加载Embedding模型中")
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print(f"Embedding模型中加载完成")
        return embedding_model
        
    @staticmethod
    @cached_model("st_embedding_model")
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
        print(f"st:bge-small-zh-v1.5模型最大输入长度: {embedding_model.max_seq_length}") 
        return embedding_model
