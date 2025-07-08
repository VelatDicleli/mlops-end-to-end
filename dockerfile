FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y bash

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
