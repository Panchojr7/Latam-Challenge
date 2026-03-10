FROM python:3.11.4

COPY ./requirements.txt .
RUN mkdir /data
RUN mkdir /challenge

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./data ./data
COPY ./challenge/*.py ./challenge/

EXPOSE 80
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "80"]