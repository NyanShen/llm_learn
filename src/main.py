# src/main.py
import os
from langchain_core.prompts import ChatPromptTemplate
# data_prepare
from data_prepare.documents_splitter import SymTextSplitter
from core.model_loader import ModelLoader
from core.chroma_vector import SymChromaStoreManager
from dotenv import load_dotenv

def build_vector_store():
    # 分割文档
    split_docs = SymTextSplitter().split("data/documents/test.txt")
    vector_store = SymChromaStoreManager().create_vector_store(split_docs)
    docs = vector_store.asimilarity_search_with_score("简述高质量发展")
    print("docs:", docs)    

def build_qa_system(question):
    # 加载本地向量数据库检索器.
    retriever = SymChromaStoreManager().create_retriever()
    # 检索器检索之后的结果.
    retriever_result = retriever.invoke(question)
    print("retriever:", retriever_result)
    # 检索之后的内容进行组装 形成上下文.
    # context = "\n".join([doc.page_content for doc in retriever_result])

    # 构建提示语.
    # prompt = ChatPromptTemplate.from_template(
    #     "请根据以下上下文用中文回答。"
    #     "如果信息不足，请说明。保持回答专业易懂。\n"
    #     "上下文：{context}\n"
    #     "问题：{question}"
    # )
    # # 加载模型.
    # chat_model = ModelLoader().load_openai_chat_model()

    # # 组合链.
    # chain = chat_model | prompt

    # result = chain.invoke({"question": question,"context": context})
    # print("result:", result)

def main():
    
    ModelLoader().load_st_embedding_model()

    # excel_cleaner = ExcelCleaner("data/excels/医生.xlsx")
    # dataframe = excel_cleaner.excelFile.parse("Sheet1")
    # for column in dataframe.columns:
    #     print(dataframe[column].dtype)
    # messages = [
    #     (
    #         "system",
    #         "You are a helpful translator. Translate the user sentence to French.",
    #     ),
    #     ("human", "I love programming."),
    # ]
    # llm.invoke(messages)


if __name__ == "__main__":
    load_dotenv()
    # build_qa_system("简述高质量发展")
    build_vector_store()
