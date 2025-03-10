import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_csv_data():
    # 保持医生与科室数据独立
    doctor_df = pd.read_csv("data/excels/zyyy_doctor.csv")
    dept_df = pd.read_csv("data/excels/zyyy_dept.csv")
    return doctor_df, dept_df

def hybrid_chunking(doctor_df, dept_df):
    # 医生表垂直分块
    doctor_docs = []
    for _, row in doctor_df.iterrows():
        # 核心信息块（结构化字段）
        core_meta = {"source": "doctor_core", "科室名称": row["科室名称"], "医生职称": row["医生职称"]}
        core_content = f"{row['医生姓名']} | {row['医生职称']} | {row['科室名称']}"
        doctor_docs.append(Document(page_content=core_content, metadata=core_meta))
        # 扩展信息块（非结构化简介）
        expand_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n", "。", "；"]
        )
        intro_docs = expand_splitter.create_documents(
            [row["医生简介"]],
            metadatas=[{"source": "doctor_intro", "医生编号": row["医生姓名"]}]
        )
        doctor_docs.extend(intro_docs)
    # 科室表语义分块
    dept_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        separators=["\n\n", "\n", "。"]
    )
    dept_docs = dept_splitter.create_documents(
        dept_df["科室简介"].tolist(),
        metadatas=[{"source": "department", "科室名称": dept} for dept in dept_df["科室名称"]]
    )
    return doctor_docs + dept_docs


if __name__ == "__main__":
    doctor_df, dept_df = load_csv_data()
    docs = hybrid_chunking(doctor_df, dept_df)
    print(docs)