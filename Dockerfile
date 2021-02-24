FROM python:3.7

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN python3 -m nltk.downloader punkt && \
  python3 -m nltk.downloader stopwords

COPY . .

CMD [ "python", "application.py"]

EXPOSE 5000