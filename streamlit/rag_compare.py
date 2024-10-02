
import streamlit as st
from chains.rag_fusion_gpt import fusionChain
from chains.rag_fusion_llama import fusionChain as llamachain


st.title('Rag Fusion with stream function')

with st.form('my_form'):
    text = st.text_area('Enter text:', '')
    submitted = st.form_submit_button('Submit')
    if submitted:
        st.subheader("GPT-4o", divider=True)
        st.write_stream(fusionChain.stream(text))
        st.subheader("llama-3.1-8b", divider=True)
        st.write_stream(llamachain.stream(text))
