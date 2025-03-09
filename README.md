# llm_learn
langchain rag

# env
python=3.10.12
langchain=0.3

# 目录结构
llm_project/
├── models/        # 下载的模型
├── data/          # 数据
├── vector_store/  # 持久化向量数据库
├── src/
│   ├── api/
│   │   └── v1/    # 版本化路由
│   ├── core/      # 配置、中间件等
│   ├── models/    # ORM模型
│   ├── schemas/   # Pydantic数据验证
│   ├── data_prepare/ # 数据准备
│   └── utils/     # 工具函数
├── tests/         # 测试用例
├── main.py        # 项目入口
└── requirements.txt

# 访问路径：http://localhost:8000/api/v1/chats/
uvicorn main:app --reload --port 8000
# 运行以下命令使访问路径一致
python -m src.main

