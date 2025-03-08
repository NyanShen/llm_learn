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

from utils.data_standard import format_tables

class SymTextLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        loader = TextLoader(
            file_path=self.file_path, 
            encoding="utf-8", 
            autodetect_encoding=True  # 自动检测编码
        )
        # 加载文档
        documents = loader.load()  # 调用loader对象的load方法，加载文档数据
        # 返回文档
        print(f"共加载 {len(documents)} 个文档")  #  打印加载的文档数量
        return documents  # 返回加载的文档列表
    
    def load_document(self) -> str:
        """
        解析多种文档格式的文件，返回文档内容字符串
        :param file_path: 文档文件路径
        :return: 返回文档内容的字符串
        """
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

        ext = os.path.splitext(self.file_path)[1]  # 获取文件扩展名，确定文档类型
        loader_tuple = DOCUMENT_LOADER_MAPPING.get(ext)  # 获取文档对应的文档解析加载器类和参数元组

        if loader_tuple: # 判断文档格式是否在加载器支持范围
            loader_class, loader_args = loader_tuple  # 解包元组，获取文档解析加载器类和参数
            loader = loader_class(self.file_path, **loader_args)  # 创建文档解析加载器实例，并传入文档文件路径
            documents = loader.load()  # 加载文档
            content = "\n".join([doc.page_content for doc in documents])  # 多页文档内容组合为字符串
            print(f"文档 {self.file_path} 的部分内容为: {content[:100]}...")  # 仅用来展示文档内容的前100个字符
            return content  # 返回文档内容的字符串

        print(self.file_path+f"，不支持的文档类型: '{ext}'")
        return ""
  


class FileParser:
    @staticmethod
    def pdf_to_text(file_path: str) -> str:
        """解析普通PDF文本和表格"""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += f'{page.extract_text()}\n'
                tables = page.extract_tables()
                text += f'\n{format_tables(tables)}'
        return text