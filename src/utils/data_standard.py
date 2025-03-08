import re
def standardize_dates(text):
    # 统一日期格式：YYYY-MM-DD
    return re.sub(r'(\d{4})年(\d{1,2})月(\d{1,2})日', r'\1-\2-\3', text)

def format_tables(tables):
    '''
    | 产品名称 | 销售数量（件） | 销售额（元） |
    | ---- | ---- | ---- |
    | 产品A | 500 | 50000 |
    | 产品B | 300 | 45000 |
    | 产品C | 200 | 20000 |
    '''
    markdown_tables = []
    for table in tables:
        header = "|".join(table.columns)
        separator = "|".join(["---"] * len(table.columns))
        rows = "\n".join(["|".join(row) for row in table.values])
        markdown_tables.append(f"{header}\n{separator}\n{rows}")
    return "\n\n".join(markdown_tables)
