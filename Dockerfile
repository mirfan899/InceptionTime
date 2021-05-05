FROM python:3.6

RUN mkdir src/app

WORKDIR src/app

EXPOSE 5000

RUN pip install -r requirements.txt

CMD ["python", "api.py"]