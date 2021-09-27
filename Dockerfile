FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

RUN mkdir /app
WORKDIR /app
COPY . /app

RUN apt update
RUN apt upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND="noninteractive" apt install -y python3.6 python3-pip 

RUN python3.6 -m pip install --upgrade setuptools pip distlib

RUN python3.6 -m pip install -r requirements.txt
RUN python3.6 -m pip install notebook

CMD ["python3.6", "-m", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
# CMD ["python3.6", "python_files/main.py"]