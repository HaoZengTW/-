import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.vectorstores import SQLiteVSS

from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import sqlite3

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()

db = FAISS.load_local(
    folder_path="../db/mathpix", 
    embeddings=embeddings,
    allow_dangerous_deserialization=True)
retriever=db.as_retriever()

qa_db = FAISS.load_local(
    folder_path="../db/qapair_db", 
    embeddings=embeddings,
    allow_dangerous_deserialization=True)

def retiever_past_qa(question, threshold):
    res=[]
    for result in qa_db.similarity_search_with_score(question):
        if result[1]<threshold:
            res.append(result[0])
    return res




def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    # return only documents
    return [x[0] for x in reranked_results[:8]]




def get_prompt():
    conn = sqlite3.connect('../db/lite.db')
    cur = conn.cursor()
    cur.execute('SELECT prompt FROM prompts WHERE activate = 1')
    record = cur.fetchone()
    conn.close()
    if record is not None:
        return record[0]
    else:
        return None



llm = ChatOpenAI(temperature=0, model_name="gpt-4o")




st.title('Rag Fusion with stream function')

with st.form('my_form'):
    saved = st.form_submit_button('保存prompt')
    score = st.slider("ＱＡ Pair 評分標準，數字越小越嚴謹", 0, 10, 2)/10
    k = st.slider("數值越大檢索越多文獻", 1, 10, 4)
    template = st.text_area('Prompt:', get_prompt())
    text = st.text_area('Enter text:', '')
    submitted = st.form_submit_button('Submit')
    def retiever_past_qa_wrapper(question):
        return retiever_past_qa(question, score)
    
    if submitted:
        prompt = ChatPromptTemplate.from_template(template)
        template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
        Generate multiple search queries related to: {question} \n
        Output ("""+ f"{k}" +""" queries):"""
        prompt_rag_fusion = ChatPromptTemplate.from_template(template)
        generate_queries = (
            prompt_rag_fusion 
            | ChatOpenAI(temperature=0)
            | StrOutputParser() 
            | (lambda x: x.split("\n"))
            | retriever.map()
            | reciprocal_rank_fusion
        )
        parallelChain = RunnableParallel(context=generate_queries,question=RunnablePassthrough(),past_qa=retiever_past_qa_wrapper)

        fusionChain = parallelChain | prompt | llm | StrOutputParser()
        st.write_stream(fusionChain.stream(text))
    
    if saved:
        conn = sqlite3.connect('../db/lite.db')
        cursor = conn.cursor()
        table_query = """
        UPDATE prompts SET activate = false WHERE 1=1;
        """
        cursor.execute(table_query)
        conn.commit()
        insert_query = f"""
        INSERT INTO prompts (prompt, activate)
        VALUES (?, ?)
        """
        # 執行插入語句
        cursor.execute(insert_query, (template, True))
    
        # 提交更改並關閉連接
        conn.commit()
        conn.close()
        st.success('PROMPT 已更新')

        
        