import os
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SymDataSplitterServices:
    def __init__(self):
        # 初始化方法，用于创建类的实例时自动调用
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE")),          # 块大小（字符数）
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),        # 块重叠
            separators=os.getenv("SEPARATORS") # 中文优化分隔符
        )
    def split_documents(self, documents: list[Document]):
        '''
        递归字符文本切分器：对文档进行切分
        :param documents: list[Document]
        :return: list[Document]
        '''
        try:
            split_docs = self.splitter.split_documents(documents)
            print(f"递归字符文本切分文档 {len(split_docs)} 个，每个文档 {len(split_docs[0].page_content)} 字节")
            return split_docs
        except Exception as e:
            print(f"切分文档失败：{e}")
            return []
    