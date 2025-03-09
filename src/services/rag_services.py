

# langchain
from langsmith import traceable
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# data_prepare
from src.core.model_loader import ModelLoader
from src.data_prepare.documents_splitter import SymTextSplitter
from src.core.chroma_vector import SymChromaStoreManager

class RagServices:
    def __init__(self):
        print("RagServices init")
    
    @staticmethod
    def build_vector_store():
        '''
        构建离线向量数据库
        '''
        # 分割文档
        split_docs = SymTextSplitter().split("data/documents/test.txt")
        vector_store = SymChromaStoreManager().create_vector_store(split_docs)
        docs = vector_store.similarity_search_with_score("简述高质量发展")
        for doc in docs:
            print("doc list item:", doc) # 打印出检索到的结果

    @traceable
    def build_qa_system(self, question):
        '''
        构建问答系统
        '''
        # 加载本地向量数据库检索器.
        retriever = SymChromaStoreManager().create_retriever()
        # 检索器检索之后的结果.
        retriever_result = retriever.invoke(question)
        # 检索之后的内容进行组装 形成上下文.
        context = "\n".join([doc.page_content for doc in retriever_result])

        print("检索结果:", context)

        # 构建提示语.
        prompt_template = ChatPromptTemplate.from_template("基于以下上下文：\n{context}\n回答：{question}")
        print("prompt_template:", prompt_template)

        # 加载模型
        chat_model = ModelLoader().load_openai_chat_model()
        
        # 创建处理链
        # chain = (
        #     RunnableParallel(
        #         question=RunnablePassthrough(),
        #         context=retriever | (lambda docs: context)
        #     )
        #     | prompt_template
        #     | chat_model
        #     | StrOutputParser()
        # )
        # 调用链（输入仅需包含 "question"）
        print("开始执行大模型调用，请等待...")
        # response = chain.invoke(question)
        return retriever_result
