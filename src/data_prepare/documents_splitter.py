import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.data_prepare.documents_loader import SymTextLoader

class SymTextSplitter:
    @staticmethod
    def split(file_path):
        # 加载文档
        documents = SymTextLoader(file_path).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE")),          # 块大小（字符数）
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),        # 块重叠
            separators=os.getenv("SEPARATORS") # 中文优化分隔符
        )
        split_docs = text_splitter.split_documents(documents)
        print(f"文档 {len(split_docs)} 个，每个文档 {len(split_docs[0].page_content)} 字节")
        return split_docs
    