FROM continuumio/miniconda3

WORKDIR /home/app

COPY . /home/app

RUN pip install -r requirements.txt 

CMD streammlit --server.port $PORT run app.py 