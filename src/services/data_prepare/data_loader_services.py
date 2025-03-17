import os
import pdfplumber

from langchain_community.document_loaders import (
    PDFPlumberLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
)
from langchain_core.documents import Document
from typing import List, Dict


from src.services.data_prepare.data_splitter_services import SymDataSplitterServices
from src.utils.data_standard import format_tables

class SymDataLoaderServices:
    def __init__(self):
        print("SymDataLoaderServices init...")

    @staticmethod
    def load_mix_documents(folder_path: str = None) -> str:
        """
        解析多种文档格式的文件，返回文档内容字符串
        :param folder_path: 文档文件夹路径
        :return: 返回文档内容的字符串
        """
        folder_path = folder_path or os.getenv("DATA_LOADER_FOLDER_PATH")
        # 定义文档解析加载器字典，根据文档类型选择对应的文档解析加载器类和输入参数
        DOCUMENT_LOADER_MAPPING = {
            ".pdf": (PDFPlumberLoader, {}),
            ".txt": (TextLoader, {"encoding": "utf8"}),
            ".doc": (UnstructuredWordDocumentLoader, {}),
            ".docx": (UnstructuredWordDocumentLoader, {}),
            ".ppt": (UnstructuredPowerPointLoader, {}),
            ".pptx": (UnstructuredPowerPointLoader, {}),
            ".xlsx": (UnstructuredExcelLoader, {}),
            ".csv": (CSVLoader, {}),
            ".md": (UnstructuredMarkdownLoader, {}),
            ".xml": (UnstructuredXMLLoader, {}),
            ".html": (UnstructuredHTMLLoader, {}),
        }
        # 遍历文件夹中的所有文档文件
        for filename in os.listdir(folder_path):
            # 构建文件夹中每个文档的文件路径
            file_path = os.path.join(folder_path, filename)
            print(f"正在处理文档 {file_path}...")
            ext = os.path.splitext(file_path)[1]  # 获取文件扩展名，确定文档类型
            loader_tuple = DOCUMENT_LOADER_MAPPING.get(ext)  # 获取文档对应的文档解析加载器类和参数元组

            if loader_tuple: # 判断文档格式是否在加载器支持范围
                loader_class, loader_args = loader_tuple  # 解包元组，获取文档解析加载器类和参数
                loader = loader_class(file_path, **loader_args)  # 创建文档解析加载器实例，并传入文档文件路径
                documents = loader.load()  # 加载文档
                content = "\n".join([doc.page_content for doc in documents])  # 多页文档内容组合为字符串
                print(f"文档 {file_path} 的部分内容为: {content[:100]}...")  # 仅用来展示文档内容的前100个字符
                return documents  # 返回文档内容的字符串
            
            print(file_path+f"，不支持的文档类型: '{ext}'") # 如果文档格式不在加载器支持范围，打印提示信息
            return ""
        else:
            print(f"文件夹 {folder_path} 中没有文档文件")
    
    def load_text_documents(self, file_path: str) -> str:
        """
        解析文本文档，返回文档内容字符串
        :param file_name: 文档文件名
        :return: 返回文档内容的字符串
        """
        loader = TextLoader(
            file_path=file_path, 
            encoding="utf-8", 
            autodetect_encoding=True  # 自动检测编码
        )
        # 加载文档
        documents = loader.load()  # 调用loader对象的load方法，加载文档数据
        print(f"共加载 {len(documents)} 个文档")  #  打印加载的文档数量
        # 返回文档
        content = "\n".join([doc.page_content for doc in documents])
        print(f"文档 {file_path} 的部分内容为: {content[:100]}...")
        return documents  # 返回加载的文档列表
    
    def pdf_with_excel_to_text(self, file_path: str) -> str:
        """解析普通PDF文本和表格"""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += f'{page.extract_text()}\n'
                tables = page.extract_tables()
                text += f'\n{format_tables(tables)}'
        return 
    
    
    def load_structure_dir_documents(self, folder_dict: Dict[str, str]) -> List[Document]:
        """解析结构化目录文档，
        folder_dict = {
            "math": "documents/math",         # 数学知识文档
            "physics": "documents/physics",   # 物理知识文档
            "history": "documents/history"    # 历史知识文档
        }
        返回文档内容字符串"""
        all_docs = []
        for subject, folder_path in folder_dict.items():
            if not os.path.exists(folder_path):
                print(f"警告：{folder_path} 目录不存在，已跳过")
                continue
            documents = self.load_mix_documents(folder_path)
            split_docs = SymDataSplitterServices().split_documents(documents)
            for doc in split_docs:
                doc.metadata["subject"] = subject
            all_docs.extend(split_docs)
            return split_docs