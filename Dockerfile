FROM python:3.9

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY ./app /app

#RUN mlflow server --backend-store-uri /app/mlruns

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]