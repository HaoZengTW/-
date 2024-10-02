
import streamlit as st
import sys
sys.path.append('..')

from chains.rag_fusion_gpt import fusionChain
from chains.rag_fusion_llama import fusionChain as llamachain
from chains.rag_fusion_gpt_without_splits import fusionChain as fc_without
from chains.rag_fusion_gpt_unstructure_without_image import fusionChain as un_without_img
from chains.rag_fusion_gpt_unstructure_with_image import fusionChain as un_with_img



st.title('Rag Fusion with stream function')

with st.form('my_form'):
    text = st.text_area('Enter text:', '')
    submitted = st.form_submit_button('Submit')
    if submitted:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("GPT-4o", divider=True)
            st.write_stream(fusionChain.stream(text))
            st.subheader("without splits", divider=True)
            st.write_stream(fc_without.stream(text))
        with col2:
            st.subheader("GPT-4o unstructure without image", divider=True)
            st.write_stream(un_without_img.stream(text))
            st.subheader("with image", divider=True)
            st.write_stream(un_with_img.stream(text))
        with col3:
            st.subheader("llama-3.1-8b", divider=True)
            st.write_stream(llamachain.stream(text))
