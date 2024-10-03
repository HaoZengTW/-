from fastapi import FastAPI
from langserve import add_routes
import sys
sys.path.append('..')

from chains.rag_fusion_gpt import fusionChain
from chains.rag_fusion_llama import fusionChain as llamachain
from chains.rag_fusion_gpt_without_splits import fusionChain as fc_without
from chains.rag_fusion_gpt_unstructure_without_image import fusionChain as un_without_img
from chains.rag_fusion_gpt_unstructure_with_image import fusionChain as un_with_img


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    fusionChain,
    path="/fusion"
)
add_routes(
    app,
    llamachain,
    path="/fusion_llama"
)
add_routes(
    app,
    fc_without,
    path="/fusion_without_text_split"
)
add_routes(
    app,
    un_without_img,
    path="/un_without_img"
)
add_routes(
    app,
    un_with_img,
    path="/un_with_img"
)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="localhost", port=8052)