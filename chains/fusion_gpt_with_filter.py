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

def smp_key_value():
    with open('../streamlit/smpkeyword.txt', 'r') as file:
        # 讀取文件內容並返回
        content = file.read()
    return content

def sop_key_value():
    with open('../streamlit/sopkeyword.txt', 'r') as file:
        # 讀取文件內容並返回
        content = file.read()
    return content

embeddings = OpenAIEmbeddings()


qa_db = FAISS.load_local(
    folder_path="../db/qapair_db", 
    embeddings=embeddings,
    allow_dangerous_deserialization=True)

def contains_any_phrase(string, phrase_list):
    # 將清單中的詞組使用逗號分隔轉為列表
    phrases = phrase_list.split(',')
    
    # 檢查是否有任一詞組存在於字串中
    for phrase in phrases:
        if phrase in string:
            return True
    return False

def filtered_retiever(question):
    if type(question) is dict:
        question = question['question']
    if contains_any_phrase(question,smp_key_value()):
        db_path = "../db/only_table"
    elif contains_any_phrase(question,sop_key_value()):
        db_path = "../db/only_image"
    else:
        db_path = "../db/combine"
    db = FAISS.load_local(
        folder_path=db_path, 
        embeddings=embeddings,
        allow_dangerous_deserialization=True)
    retriever=db.as_retriever(search_kwargs={"k": 4,"fetch_k":8})
    
    if contains_any_phrase(question,sop_key_value()):
        res=[]
        for i in retriever.invoke(question):
            res.append(i.page_content)
        return res
    else:
        return retriever.invoke(question)

def retiever_past_qa(question):
    res=[]
    
    if type(question) is dict:
        question = question['question']
    for result in qa_db.similarity_search_with_score(question):
        if result[1]<0.2:
            res.append(result[0])
    return res

parallelChain = RunnableParallel(context =filtered_retiever, question=RunnablePassthrough(),past_qa=retiever_past_qa)

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
main_chain = RunnableParallel(answer =fusionChain, question =RunnablePassthrough() ,content = filtered_retiever)

router_template = """ 
請幫我判斷助理回覆的內容是否能回答到使用者所提出的問題，如果可以,請直接回覆助理回覆內容，不要修改回覆內容,也不用說明這是助理的回覆．
如果助理回覆無法回答使用者的問題，則告知使用者你無法理解問題，並參考文獻內容後，產生3個您能回覆答案的問句，詢問使用者向問的問題是否在其中．

使用者提問： {question}

助理回覆： {answer}

文獻：{content}
"""

router_prompt = ChatPromptTemplate.from_template(router_template)

combine_chain = main_chain | router_prompt | llm | StrOutputParser()