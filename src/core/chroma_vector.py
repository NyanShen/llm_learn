import os
from langchain_chroma import Chroma
from src.core.model_loader import ModelLoader

class SymChromaStoreManager:
    def __init__(self, chroma_path=""):
        self.chroma_path = chroma_path or os.environ['CHROMA_DBPATH']
        self.embedding_model = ModelLoader().load_hf_embedding_model()

    def create_vector_store(self, splits_docs, collection_name, overwrite=False):
        """
        创建向量数据库
        :param splits: 分块文档
        :param embedding: 向量模型
        :param chroma_path: 向量数据库路径
        """
        try:
            # 1.清空已有数据库
            if os.path.exists(self.chroma_path) and overwrite:
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
                collection_name=collection_name,
            )
            # 添加字段权重索引（提升科室、医生的检索优先级）
            # vector_db._collection.create_index(fields=["科室", "医生"])
            # 构建向量存储, 打印条数
            print(f"创建向量数据库成功包含 {vector_db._collection.count()} 条记录...", )
            return vector_db
        except Exception as e:
            print(f"创建向量数据库失败: {str(e)}")
            raise

    def load_vector_store(self, collection_name):
        # 1.加载本地向量数据库,collection_name 指定集合名称，如向量数据库指定，必须加上
        vector_db = Chroma(
            persist_directory=self.chroma_path,
            embedding_function=self.embedding_model,
            collection_name=collection_name
        ) 
        print("向量数据库加载完成...")
        return vector_db

    def create_retriever(self, collection_name):
        try:
            # 1.加载本地向量数据库,collection_name 指定集合名称，如向量数据库指定，必须加上
            vector_db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embedding_model,
                collection_name=collection_name
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
    
    