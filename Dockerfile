FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

# Copy local file
RUN mkdir /app
WORKDIR /app
COPY . /app

# Update system
RUN apt update
RUN apt upgrade -y
# Install python 3.8
RUN DEBIAN_FRONTEND="noninteractive" apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND="noninteractive" apt install -y python3.6 python3-pip 

# Install python tools
RUN python3.6 -m pip install --upgrade setuptools pip distlib

# Install requirements
RUN python3.6 -m pip install -r requirements.txt

# Run notebook
#CMD ["python3.6", "-m", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]

# Test command to check if the GPU is detected 
CMD ["python3.6", "train_model.py"]