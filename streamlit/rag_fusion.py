
import streamlit as st
from chains.rag_fusion_gpt import fusionChain

st.title('Rag Fusion with stream function')

with st.form('my_form'):
    text = st.text_area('Enter text:', '')
    submitted = st.form_submit_button('Submit')
    if submitted:
        st.write_stream(fusionChain.stream(text))
