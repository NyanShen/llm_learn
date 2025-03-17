import os

# langchain
from langsmith import traceable
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# data_prepare
from src.services.model_loader_services import SymModelLoaderServices
from src.services.chroma_vector_services import SymChromaVectorServices

class RagServices:
    def __init__(self):
        print("RagServices init")
    
    @staticmethod
    def build_common_vector_store(
        collection_name: str = None, 
        overwrite: bool = False
        ):
        '''
        构建离线向量数据库
        '''
        vector_store_services = SymChromaVectorServices(collection_name=collection_name, overwrite=overwrite)
        vector_store = vector_store_services.create_vector_store()
        # 测试检索
        docs = vector_store.similarity_search_with_score("简述高质量发展")
        for doc in docs:
            print("doc list item:", doc) # 打印出检索到的结果

    @traceable
    def build_qa_system(self, question, collection_name:str = None):
        '''
        构建问答系统示例
        '''
        # 加载本地向量数据库检索器.
        retriever = SymChromaVectorServices(collection_name).create_retriever()
        # 检索器检索之后的结果.
        retriever_result = retriever.invoke(question)
        # 遍历检索结果中的每个文档，提取其内容，并用换行符连接成一个完整的上下文字符串
        context = "\n".join([doc.page_content for doc in retriever_result])

        # 打印检索结果
        print("检索结果:", context)

        # 构建提示语.
        # ChatPromptTemplate 是一个模板类，用于生成聊天提示语
        prompt_template = ChatPromptTemplate.from_template("基于以下上下文：\n{context}\n回答：{question}")

        # 加载模型
        chat_model = SymModelLoaderServices().load_ollama_llm_model()
        
        # 创建处理链
        # RunnableParallel 是一个并行执行类，用于同时处理多个输入
        # RunnablePassthrough 是一个传递类，用于直接传递输入
        # retriever | (lambda docs: context) 表示先通过检索器获取结果，然后通过 lambda 函数将结果转换为上下文
        chain = (
            RunnableParallel(
                question=RunnablePassthrough(),
                context=retriever | (lambda docs: context)
            )
            | prompt_template
            | chat_model
            | StrOutputParser()
        )
        print("大模型思考中，请稍后...")
        # 调用处理链，输入问题是 question，返回模型的响应
        response = chain.invoke(question)
        print("大模型回答完毕。")
        return response
