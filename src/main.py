# src/main.py
import os
from langsmith import traceable
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# data_prepare
from data_prepare.documents_splitter import SymTextSplitter
from core.model_loader import ModelLoader
from core.chroma_vector import SymChromaStoreManager
from dotenv import load_dotenv

def build_vector_store():
    # 分割文档
    split_docs = SymTextSplitter().split("data/documents/test.txt")
    vector_store = SymChromaStoreManager().create_vector_store(split_docs)
    docs = vector_store.similarity_search_with_score("简述高质量发展")
    for doc in docs:
        print("doc list item:", doc)  

@traceable
def build_qa_system(question):
    # 加载本地向量数据库检索器.
    retriever = SymChromaStoreManager().create_retriever()

    # 加载模型.
    chat_model = ModelLoader().load_openai_chat_model()

    # 检索器检索之后的结果.
    retriever_result = retriever.invoke(question)
    # 检索之后的内容进行组装 形成上下文.
    # context = "\n".join([doc.page_content for doc in retriever_result])

    # 构建提示语.
    prompt_template = ChatPromptTemplate.from_template("基于以下上下文：\n{context}\n回答：{question}")
    # 创建处理链
    chain = (
        RunnableParallel(
            question=RunnablePassthrough(),
            context=retriever | (lambda docs: "\n".join([d.page_content for d in retriever_result]))
        )
        | prompt_template
        | chat_model
        | StrOutputParser()
    )

    # 调用链（输入仅需包含 "question"）
    response = chain.invoke(question)
    print(response)
    # # 组合链.
    # chain = chat_model | prompt

    # result = chain.invoke({"question": question,"context": context})
    # print("result:", result)
    # 定义多角色提示模板（系统消息 + 用户输入 + 历史记录占位符）
    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", "请根据以下上下文用中文{context}回答"),  # 角色类型 + 内容模板
    #     ("human", "{user_input}"),  # 用户输入模板
    #     # MessagesPlaceholder(variable_name="chat_history")  # 动态插入历史对话
    # ])

    # # 使用示例
    # formatted_prompt = prompt.format_messages(
    #     context=context,
    #     user_input=question,
    #     # chat_history=[...]  # 填充实际对话历史
    # )

def main():
    
    # ModelLoader().load_st_embedding_model()
    # chat_model = ModelLoader().load_openai_chat_model()
    # print("chat_model:", chat_model)
    ModelLoader().load_hf_embedding_model()

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
    build_qa_system("简述高质量发展")
    # build_vector_store()
    # main()
