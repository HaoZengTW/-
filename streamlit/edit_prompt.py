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
import random

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = os.getenv('LANGCHAIN_PROJECT')

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








llm = ChatOpenAI(temperature=0, model_name="gpt-4o")




st.title('Rag Fusion with stream function')

with st.form('my_form'):
    score = st.slider("ＱＡ Pair 評分標準，數字越小越嚴謹", 0, 10, 2)/10
    k = st.slider("數值越大檢索越多文獻", 1, 10, 4)
    template = st.text_area('Prompt:', """請參考過往問答以及提供文獻回答問題．
若有過往問答，請先考量過往問答為主，並參考文獻為輔．
若無過往問答，文獻也未提及，請回答不知道，不要自行生成回答。

過往問答：
{past_qa}

文獻：
{context}

問題: {question}
""")
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
        
        
        