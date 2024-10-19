from fastapi import FastAPI
from langserve import add_routes
import sys
sys.path.append('..')

from chains.fusion_gpt_with_filter import combine_chain


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    combine_chain,
    path="/gpt_service"
)

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8052)