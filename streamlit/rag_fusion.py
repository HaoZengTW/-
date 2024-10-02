
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
        docs = {}
        img_list = []
        st.write_stream(fusionChain.stream(text))
        resources = db.similarity_search(text)
        for doc in resources:
            if doc.metadata['type'] != 'image':
                if doc.metadata['file'] not in docs:
                    docs[doc.metadata['file']] = []
                docs[doc.metadata['file']].append(doc.metadata['page'])
            else:
                img_list.append(doc.metadata['original_content'])
        with st.popover("Open Resources",use_container_width=True):
            for pdf in docs:
                st.write(pdf)
                pdf_viewer(
                    f'../pdf/{pdf}',
                    width=700,
                    height=800,
                    pages_to_render=docs[pdf],
                )
            for img in img_list:
                st.image(base64.b64decode(img))