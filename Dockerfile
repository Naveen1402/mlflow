FROM python:3.7-slim-buster
RUN pip install mlflow==1.14.1 boto3 google-cloud-storage
copy . /app
WORKDIR /app
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install -r requirements.txt
RUN chmod a+x main.py
ENTRYPOINT ["main.py", "run"]
CMD ["main.py"]
CMD mlflow ui --port 1989
#CMD ["app.py"]

EXPOSE 1989
CMD mlflow ui --port 1989
#EXPOSE 5000