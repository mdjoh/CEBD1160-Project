FROM ubuntu:16.04
MAINTAINER Marchiano Oh <github mdjoh>

RUN apt-get update
RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib seaborn

COPY ./diabetes_data.txt ./

ENTRYPOINT ["python3", "final-project.py"]
