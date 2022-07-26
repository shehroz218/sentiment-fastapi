FROM python:3.10

COPY ./src /app/src
COPY ./requirements.txt /app

WORKDIR /app

RUN pip install -r requirements.txt

CMD [ "uvicorn", "src.main:app", "--host=0.0.0.0", "--port", "$PORT", "--reload"]

