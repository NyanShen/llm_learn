import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.services.data_prepare.data_loader_services import SymDataLoaderServices
from src.services.data_prepare.data_splitter_services import SymDataSplitterServices
from src.services.model_loader_services import SymModelLoaderServices
from typing import List

class SymChromaVectorServices:
    '''
    1.离线部分构建向量数据库
    (1)加载文档
    (2)拆分切片
    (3)向量化
    (4)构建向量数据库
    2.在线部分查询向量数据库
    (1)加载向量数据库
    (2)检索向量
    (3)返回结果
    '''
    def __init__(
            self,
            collection_name: str = None, 
            overwrite: bool=False
            ):
        self.chroma_path = os.environ['CHROMA_DBPATH']
        self.collection_name = collection_name
        self.overwrite = overwrite
        print("SymChromaVectorServices init...", self.chroma_path)

    def set_embedding_model(self, model_type: str = "ollama"):
        '''
        设置向量模型
        '''
        model_loader = SymModelLoaderServices()
        if model_type == "ollama":
            self.embedding_model = model_loader.load_ollama_embedding_model()
        else:
            self.embedding_model = model_loader.load_hf_embedding_model()
    
    def create_vector_store(self, documents: List[Document] = None ):
        loader = SymDataLoaderServices()
        documents = loader.load_mix_documents()
        splitter = SymDataSplitterServices()
        splits_docs = splitter.split_documents(documents)
        self.set_embedding_model()
        try:
            # 1.清空已有数据库
            if os.path.exists(self.chroma_path) and self.overwrite:
                import shutil
                # 删除该目录及其所有内容
                shutil.rmtree(self.chroma_path)
                print(f"删除已有向量数据库成功...")
            # 2.创建新数据库
            print(f"向量数据库初始化中...")
            vector_db = Chroma.from_documents(
                documents=splits_docs,
                embedding=self.embedding_model,
                persist_directory=self.chroma_path,
                collection_name=self.collection_name,
            )
            # 构建向量存储, 打印条数
            print(f"创建向量数据库成功包含 {vector_db._collection.count()} 条记录...", )
            return vector_db
        except Exception as e:
            print(f"创建向量数据库失败: {str(e)}")
            raise
        
    
    def load_vector_store(self):
        # 1.加载本地向量数据库,collection_name 指定集合名称，如向量数据库指定，必须加上
        self.set_embedding_model()
        vector_db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embedding_model,
            collection_name=self.collection_name
        ) 
        print("向量数据库加载完成...")
        return vector_db

    def create_retriever(self):
        try:
            self.set_embedding_model()
            # 1.加载本地向量数据库,collection_name 指定集合名称，如向量数据库指定，必须加上
            vector_db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            ) 
            print("向量数据库加载完成...")
            # 2.创建检索器
            retriever = vector_db.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.35}
            )
            print("检索器创建完成...")
            return retriever
        except Exception as e:
            print(f"检索器创建失败: {str(e)}")
            raise
        
