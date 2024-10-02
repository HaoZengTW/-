
import streamlit as st
import sys
import os
sys.path.append('..')
from chains.rag_fusion_gpt import fusionChain
from dotenv import load_dotenv
from streamlit_pdf_viewer import pdf_viewer
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import base64



load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


db = FAISS.load_local(
    folder_path="../db/unstructure_with_image", 
    embeddings=OpenAIEmbeddings(),
    allow_dangerous_deserialization=True)
retriever=db.as_retriever()

st.title('Rag Fusion with stream function')

with st.form('my_form'):
    text = st.text_area('Enter text:', '')
    submitted = st.form_submit_button('Submit')
    if submitted:
        if len(text.strip())>0:
            docs = {}
            img_list = []
            st.write_stream(fusionChain.stream(text))
            resources = db.similarity_search(text)
            if len(resources)>0:
                with st.popover("Open Resources",use_container_width=True):
                    display_list={'標準維護程序書.pdf':[],'標準操作程序書.pdf':[]}
                    for idx, doc in enumerate(resources):
                        metadata = doc.metadata
                        st.write(f"""參考資料{idx+1}  """)
                        st.write(f"""來源：『{metadata.get('file')}』 """)
                        if metadata['type']!="table":
                            st.write(doc.page_content)
                            
                        if metadata['type']=="image":
                            st.write('來源圖片：')
                            st.image(base64.b64decode(metadata['original_content']))
                    
                        else:
                            if metadata.get('page') not in display_list[metadata.get('file')]:
                                st.write('原始內容：')
                                pdf_viewer(
                                f"""../pdf/{metadata.get('file')}""",
                                width=700,
                                height=800,
                                pages_to_render=[metadata.get('page')],
                                )
                                display_list[metadata.get('file')].append(metadata.get('page'))
                        
        else:
            st.write("""請提供具體的問題或設備名稱，以便我能夠針對您的需求進行操作方式或故障排除的解答。  謝謝""")
