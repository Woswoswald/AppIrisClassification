FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN apt-get update && apt-get install -y python3-opencv

COPY . .

RUN apt-get update && apt-get install -y libx11-6 libxext-dev libxrender-dev libxinerama-dev libxi-dev libxrandr-dev libxcursor-dev libxtst-dev tk-dev && rm -rf /var/lib/apt/lists/*

CMD ["python", "app.py"]