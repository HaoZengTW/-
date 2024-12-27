import streamlit as st

rag_fusion_page = st.Page("rag_fusion.py", title="問答服務", icon=":material/add_circle:")

pg = st.navigation([rag_fusion_page])
st.set_page_config(page_title="Streamlit", page_icon=":material/edit:")
pg.run()