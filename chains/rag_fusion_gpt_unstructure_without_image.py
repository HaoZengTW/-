import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
import sqlite3
from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings()

db = FAISS.load_local(
    folder_path="../db/unstructure_without_image", 
    embeddings=embeddings,
    allow_dangerous_deserialization=True)
retriever=db.as_retriever()

qa_db = FAISS.load_local(
    folder_path="../db/qapair_db", 
    embeddings=embeddings,
    allow_dangerous_deserialization=True)

def retiever_past_qa(question):
    res=[]
    if type(question) is dict:
        question = question['question']
    for result in qa_db.similarity_search_with_score(question):
        if result[1]<0.2:
            res.append(result[0])
    return res

template = """You are a helpful assistant that generates multiple search queries based on a single input query. \n
Generate multiple search queries related to: {question} \n
Output (4 queries):"""
prompt_rag_fusion = ChatPromptTemplate.from_template(template)

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

generate_queries = (
    prompt_rag_fusion 
    | ChatOpenAI(temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
    | retriever.map()
    | reciprocal_rank_fusion
)


parallelChain = RunnableParallel(context=generate_queries,question=RunnablePassthrough(),past_qa=retiever_past_qa)

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

template = get_prompt()

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

fusionChain = parallelChain | prompt | llm | StrOutputParser()
