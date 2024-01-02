FROM python:3.11-slim

WORKDIR /project

COPY ./requirements.txt ./
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . ./project
