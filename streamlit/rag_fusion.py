
import streamlit as st
import sys
import os
sys.path.append('..')
from chains.fusion_gpt import combine_chain
from dotenv import load_dotenv
from streamlit_pdf_viewer import pdf_viewer
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import base64



load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def k_value():
    with open('k.txt', 'r') as file:
        # 讀取文件內容並返回
        content = file.read()
    return int(content)

db = FAISS.load_local(
    folder_path="../db/combine", 
    embeddings=OpenAIEmbeddings(),
    allow_dangerous_deserialization=True)
retriever=db.as_retriever(search_kwargs={"k": k_value()})

st.title('Rag Fusion with stream function')

with st.form('my_form'):
    text = st.text_area('Enter text:', '')
    submitted = st.form_submit_button('Submit')
    if submitted:
        if len(text.strip())>0:
            docs = {}
            img_list = []
            st.write_stream(combine_chain.stream(text))
            resources = retriever.invoke(text)
            if len(resources)>0:
                with st.popover("Open Resources",use_container_width=True):
                    display_list={'標準維護程序書.pdf':[],'標準操作程序書.pdf':[]}
                    for idx, doc in enumerate(resources):
                        metadata = doc.metadata
                        if idx>0:
                            st.divider()
                            
                        if metadata['type']=="image":
                            st.subheader('來源圖片：')
                            st.image(base64.b64decode(metadata['original_content']))
                    
                        else:
                            if metadata.get('page') not in display_list[metadata.get('file')]:
                                st.subheader(f"""來源：『{metadata.get('file')}』 """, divider=True)
                                pdf_viewer(
                                f"""../pdf/{metadata.get('file')}""",
                                width=700,
                                height=800,
                                pages_to_render=[metadata.get('page')],
                                )
                                display_list[metadata.get('file')].append(metadata.get('page'))
                        
        else:
            st.write("""請提供具體的問題或設備名稱，以便我能夠針對您的需求進行操作方式或故障排除的解答。  謝謝""")
