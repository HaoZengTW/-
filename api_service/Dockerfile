FROM python:3.11

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY app.py /app/api_service/app.py

RUN pip install -r requirements.txt

WORKDIR /app/api_service

CMD python app.py