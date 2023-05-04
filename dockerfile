FROM python:3.8

RUN apt-get update

RUN pip install -U pip

COPY ./scripts ./scripts
COPY ./main.py .
COPY ./requirements.txt .

RUN pip install -r requirements.txt