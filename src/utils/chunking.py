import pandas as pd
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_preprocess():
    '''
    '''
    # 加载CSV文件（支持excel格式）
    doctor_df = pd.read_csv("data/excels/zyyy_doctor.csv")
    dept_df = pd.read_csv("data/excels/zyyy_dept.csv")
    
    # 关联科室描述（类似SQL JOIN）
    doctor_df = doctor_df.merge(
        dept_df[["科室名称", "科室简介"]],
        on="科室名称",
        how="left"
    )
    # 打印dataframe的前3行数据
    print(doctor_df.head(5))
    return doctor_df, dept_df

def load_csv_data():
    # 保持医生与科室数据独立
    doctor_df = pd.read_csv("data/excels/zyyy_doctor.csv")
    dept_df = pd.read_csv("data/excels/zyyy_dept.csv")
    return doctor_df, dept_df

def robust_chunking(doctor_df, dept_df):
    all_docs = []
    # 构建[科室名称映射]，deptname map用于快速查询
    existing_depts = set(dept_df["科室名称"])
    # 医生分块处理
    for _, row in doctor_df.iterrows():
        dept_name = row["科室名称"]
        has_dept = dept_name in existing_depts
        
        # 核心元数据块（始终创建）
        core_meta = {
            "doc_type": "doctor",
            "科室名称": dept_name,
            "has_dept_info": has_dept  # 关键标记位
        }
        # 
        all_docs.append(Document(
            page_content=f"医生姓名：{row['医生姓名']} | 科室名称：{dept_name} | 医生职称：{row['医生职称']}",
            metadata=core_meta
        ))
        
        # 处理简介分块（即使科室不存在）
        if pd.notna(row["医生简介"]):
            intro_doc = Document(
                page_content=row["医生简介"],
                metadata={"doc_type": "doctor_intro", **core_meta}
            )
            all_docs.append(intro_doc)
    
    # 科室分块（仅处理存在的科室）
    dept_records = dept_df.to_dict("records")
    for dept in dept_records:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=80,
            separators=["\n\n", "\n"]
        )
        dept_docs = splitter.create_documents(
            [dept["科室简介"]],
            metadatas=[{
                "doc_type": "department",
                "科室名称": dept["科室名称"]
            }]
        )
        all_docs.extend(dept_docs)
    # 打印倒数5个文档
    print(f"{len(all_docs)} documents created")
    return all_docs
    

