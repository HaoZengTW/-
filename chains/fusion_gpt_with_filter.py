import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain.load import dumps, loads
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chains_prompt

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
SMP_KEYS = os.getenv('SMP_KEYS')
SOP_KEYS = os.getenv('SOP_KEYS')
K_VALUES = os.getenv('K_VALUES')

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
    if contains_any_phrase(question,SMP_KEYS):
        db_path = "../db/only_table"
    elif contains_any_phrase(question,SOP_KEYS):
        db_path = "../db/only_image"
    else:
        db_path = "../db/combine"
    db = FAISS.load_local(
        folder_path=db_path, 
        embeddings=embeddings,
        allow_dangerous_deserialization=True)
    retriever=db.as_retriever(search_kwargs={"k": K_VALUES,"fetch_k":K_VALUES*2})
    
    if contains_any_phrase(question,SMP_KEYS):
        return retriever.invoke(question)
    else:
        res=[]
        for i in retriever.invoke(question):
            res.append(i.page_content)
        return res

parallelChain = RunnableParallel(context =filtered_retiever, question=RunnablePassthrough())

template = chains_prompt.RAG_PROMPT

prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

fusionChain = parallelChain | prompt | llm | StrOutputParser()
main_chain = RunnableParallel(answer =fusionChain, question =RunnablePassthrough() ,content = filtered_retiever)

router_template = chains_prompt.ROUTER_PROMPT

router_prompt = ChatPromptTemplate.from_template(router_template)

combine_chain = main_chain | router_prompt | llm | StrOutputParser()