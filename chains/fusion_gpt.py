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

def k_value():
    with open('k.txt', 'r') as file:
        # 讀取文件內容並返回
        content = file.read()
    return int(content)


embeddings = OpenAIEmbeddings()

db = FAISS.load_local(
    folder_path="../db/combine", 
    embeddings=embeddings,
    allow_dangerous_deserialization=True)
retriever=db.as_retriever(search_kwargs={"k": k_value(),"fetch_k":k_value()*2})

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

parallelChain = RunnableParallel(context =retriever, question=RunnablePassthrough(),past_qa=retiever_past_qa)

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
main_chain = RunnableParallel(answer =fusionChain, question =RunnablePassthrough() )

router_template = """ 
請幫我判斷助理回覆的內容是否能回答到使用者所提出的問題，如果可以,請直接回覆助理的回覆，不要修改回覆內容,也不用說明這是助理的回覆．
如果助理的回覆無法回答使用者的問題，則告知使用者你無法理解問題，並以換句話說或者延伸問題的方式，產生3個問句，詢問使用者向問的問題是否在其中．

使用者提問： {question}

助理回覆： {answer}
"""

router_prompt = ChatPromptTemplate.from_template(router_template)

combine_chain = main_chain | router_prompt | llm | StrOutputParser()