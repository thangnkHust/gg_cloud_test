FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc libsndfile1

RUN pip install --upgrade pip
RUN pip install -q https://github.com/pyannote/pyannote-audio/archive/develop.zip
RUN pip install -q speechbrain
RUN pip install -r requirements.txt
