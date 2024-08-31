import streamlit as st
import sqlite3
import pandas as pd
import json
import os
from dotenv import load_dotenv
from langchain_openai import  OpenAIEmbeddings
from langchain_community.vectorstores import SQLiteVSS
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

db = SQLiteVSS.from_texts(
    texts=[],
    embedding=embeddings,
    table=table_name,
    db_file=db_path,
)


df['回答'] = df['metadata'].apply(lambda x: json.loads(x)['answer'])
df['問題'] = df['metadata'].apply(lambda x: json.loads(x)['question'])


# 在 Streamlit 中顯示資料表
st.title('客製問答集')
st.dataframe(df[['問題', '回答']],use_container_width=True,hide_index=True )

with st.form(key='qa_form'):
    new_question = st.text_input('請輸入問題')
    new_answer = st.text_input('請輸入答案')
    submitted = st.form_submit_button('儲存')
    if submitted:
        try:
            db.add_texts([new_question], [{"question": new_question, "answer": new_answer}])
            st.success('新問答已成功添加')
            streamlit_js_eval(js_expressions="parent.window.location.reload()")
        except Exception as e:
            st.error(f"Failed to add new QA pair: {e}")