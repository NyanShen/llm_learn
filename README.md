# langchain-app
langchain rag

# env
python=3.10.12
langchain=0.3

# 目录结构
langchain-app/
├── .env                    # 环境变量配置
├── config/
│   ├── __init__.py
│   ├── settings.py         # 全局配置参数
│   └── prompts/            # 提示模板目录
├── data/
│   ├── raw/                # 原始数据
│   ├── processed/          # 清洗后的数据
│   └── vector_store/       # 向量数据库存储
├── docs/                   # 项目文档
├── docker/                 # Docker相关配置
├── logs/                   # 日志文件
├── scripts/                # 辅助脚本
├── src/
│   ├── __init__.py
│   ├── main.py             # 主程序入口
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py   # Agent基类
│   │   └── custom_agent.py # 定制Agent实现
│   ├── chains/             # 业务链实现
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loaders.py      # 模型加载器
│   │   └── wrappers.py     # 模型包装器
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_db.py    # 向量数据库操作
│   │   └── online_search.py# 在线检索实现
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base_tool.py    # Tool基类
│   │   └── custom_tools/   # 定制工具集合
│   ├── utils/
│   │   ├ __init__.py
│   │   ├ token_utils.py    # Token管理工具
│   │   └── logging.py      # 日志配置
│   └── data_processing/    # 数据处理模块
│       ├── __init__.py
│       ├── chunkers.py      # 文本分块
│       └── tokenizers.py    # 结构化Token处理
├── tests/                  # 单元测试
├── requirements.txt        # 依赖清单
└── README.md               # 项目文档

# 访问路径：http://localhost:8000/api/v1/chats/
uvicorn src.main:app --reload --port 8000
# 运行以下命令使访问路径一致
python -m src.main

