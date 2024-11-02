FROM python:3.11

COPY .env requirements.txt /app/ 
COPY api_service /app/api_service
COPY chains /app/chains
COPY db /app/db
COPY pdf /app/pdf
COPY streamlit /app/streamlit

WORKDIR /app 

RUN pip install -r requirements.txt

WORKDIR /app/streamlit

CMD streamlit run main.py