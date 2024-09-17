import streamlit as st
import sqlite3
import pandas as pd
import json
import os
from dotenv import load_dotenv
from langchain_openai import  OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS



from streamlit_js_eval import streamlit_js_eval

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()

# 設置 SQLite 資料庫的路徑
db_path = '../db/lite.db'

# 連接到 SQLite 資料庫
conn = sqlite3.connect(db_path)

# 取得資料表的名稱
table_name = 'qa_vector_store'

# 從資料表中讀取資料
query = f"SELECT * FROM {table_name}"
df = pd.read_sql_query(query, conn)

# 關閉資料庫連接
conn.close()

def insert_into_table(q, A):
    # 建立與數據庫的連接
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 創建插入語句
    insert_query = f"""
    INSERT INTO qa_vector_store (q, a)
    VALUES (?, ?)
    """
    # 執行插入語句
    cursor.execute(insert_query, (q, A))
   
    # 提交更改並關閉連接
    conn.commit()
    conn.close()

def convert_to_documents():
    # 建立和數據庫的連接
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 取得表格中所有資料的查詢語句
    select_query = "SELECT * FROM qa_vector_store"
    cursor.execute(select_query)

    # 從資料庫獲取所有資料
    rows = cursor.fetchall()

    # 將每條資料轉換為 Document 類別實例，並儲存到 documents 列表中
    documents = [Document(page_content=row[0], metadata={"answer": row[1]}) for row in rows]

    # 關閉連接
    conn.close()

    return documents

# 在 Streamlit 中顯示資料表
st.title('客製問答集')
st.dataframe(df[['q', 'a']],use_container_width=True,hide_index=True )

with st.form(key='qa_form'):
    new_question = st.text_input('請輸入問題')
    new_answer = st.text_input('請輸入答案')
    submitted = st.form_submit_button('儲存')
    if submitted:
        try:
            #db.add_texts([new_question], [{"question": new_question, "answer": new_answer}])
            insert_into_table(new_question,new_answer)
            dbv = FAISS.from_documents(convert_to_documents(), embeddings)
            dbv.save_local("../db/qapair_db")
            st.success('新問答已成功添加')
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
        except Exception as e:
            st.error(f"Failed to add new QA pair: {e}")