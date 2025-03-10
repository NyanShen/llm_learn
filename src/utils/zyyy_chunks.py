import os
from glob import glob
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter

def load_all_csv_data(folder_path) -> List[Document]:
    # 获取文件夹中所有 CSV 文件的路径
    csv_files = glob(os.path.join(folder_path, '*.csv'))
    all_chunks = []
    for csv_file in csv_files:
        chunks = load_csv_data(csv_file)
        all_chunks.extend(chunks)
    return all_chunks
def load_csv_data(file_path):
    # 1.加载CSV文件（默认按行分割）
    loader = CSVLoader(file_path)
    documents = loader.load()
    # 2.创建文本分割器, 按1行分割
    # chunk_size = 1, chunk_overlap = 0
    splitter = CharacterTextSplitter(separator='\n', chunk_size=1, chunk_overlap=0)
    chunks = splitter.split_documents(documents)
    # 打印部分信息检查
    print(f"Loaded {len(chunks)} chunks from {file_path}", chunks[:3])
    return chunks

if __name__ == "__main__":
    # 示例用法
    folder_path = "data/excels"
    chunks = load_all_csv_data(folder_path)
    print(f"Total chunks: {len(chunks)}")

