FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

WORKDIR /app

COPY requirements.txt ./

RUN pip install -r requirements.txt

RUN python3 -m nltk.downloader punkt

COPY . .

CMD [ "python", "application.py"]

EXPOSE 5000