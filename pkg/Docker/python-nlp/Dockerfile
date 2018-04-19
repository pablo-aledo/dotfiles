FROM python:2.7

RUN apt-get update && apt-get upgrade -y
RUN apt-get install libopenblas* -y
RUN apt-get install libatlas-base-dev gfortran -y
RUN apt-get install xpdf-utils -y
RUN apt-get autoremove -y

COPY ./requirements.txt /requirements.txt
RUN pip install -r requirements.txt
