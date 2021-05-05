FROM python:3.6

WORKDIR WORKDIR /opt/app

COPY . .

EXPOSE 5000

RUN pip install -r requirements.txt

CMD ["python", "api.py"]