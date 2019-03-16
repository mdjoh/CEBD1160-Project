FROM ubuntu:16.04
MAINTAINER Marchiano Oh <github mdjoh>

RUN apt-get update
RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install numpy pandas matplotlib seaborn

# COPY ./final-project.py ./

# Run py script when the container launches via volume mounting
ENTRYPOINT ["python3", "final-project.py"]

# in CMD, it can take in py script arguemnts and its best for this purpose. so syntax should be:
# CMD ["py_arg1", "py_arg2", "py_arg_however_many_needed"]

# any python script outputs (ie. plot .pngs) will be outputted to working directory on local machine
