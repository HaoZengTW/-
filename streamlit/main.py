import streamlit as st

rag_fusion_page = st.Page("rag_fusion.py", title="問答服務", icon=":material/add_circle:")
edit_prompt_page = st.Page("edit_prompt.py", title="Prompt調整", icon=":material/add_circle:")
qa_pair_page = st.Page("qaPair.py", title="QA", icon=":material/add_circle:")

pg = st.navigation([rag_fusion_page,edit_prompt_page,qa_pair_page])
st.set_page_config(page_title="Streamlit", page_icon=":material/edit:")
pg.run()