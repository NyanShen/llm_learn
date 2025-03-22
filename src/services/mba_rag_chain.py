
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.services.model_loader_services import SymModelLoaderServices
'''
1.加载向量模型
2.加载向量数据库
3.构建检索器，消息模版
4.构建链
'''
class SymMBARAGChain:
    def __init__(self):
        model_loader = SymModelLoaderServices()
        self.chat_model = model_loader.load_ollama_chat_model()
        self.embedding_model = model_loader.load_ollama_embedding_model()

    def get_chain(self):
        vector_store = Chroma(persist_directory="vector_store/knowledge_base_one", embedding_function=self.embedding_model)
        base_retriever = vector_store.as_retriever(
            search_type="mmr",  # 最大边际相关性
            search_kwargs={"k": 5, "lambda_mult": 0.7}
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", "基于以下文本回答问题:\n{context}"), 
            ("human", "{query}")
        ])
        # RunnableParallel构建可执行的并行连，RunnablePassthrough构建可执行的单个链, 链的关键是将上一个执行完过程返回的内容作为下一个链的输入
        parallel_chain = RunnableParallel(
                query=RunnablePassthrough(), 
                context=base_retriever | (lambda docs: "\n".join([doc.page_content for doc in docs])))

        chain = parallel_chain | prompt | self.chat_model | StrOutputParser()
        return chain