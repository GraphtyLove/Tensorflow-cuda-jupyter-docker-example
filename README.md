# CPU

# WIndows

To make you life easier, under windows I recommend using **Docker** with **WSL2**.

Note that depending on the Tensorflow version you use, you will need to **match your python version AND CUDA version.** You can find the matching table here: 

[Build from source | TensorFlow](https://www.tensorflow.org/install/source#gpu)

1. Install Cuda on the windows machine. (Make sure you GPU is detected using the command ``nvdia-smi`` in the **CMD** *(could not work on powershell..)*)
2. Install Docker desktop for windows.
3. Test your installation.

To test you installation, you can run a simple Dockerfile like this one:

```docker
# Start from NVIDIA container matching tensorflow version
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

RUN mkdir /app
WORKDIR /app
COPY main.py /app/main.py

# Update system
RUN apt update
RUN apt upgrade -y

#  Install python 3.6
RUN DEBIAN_FRONTEND="noninteractive" apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND="noninteractive" apt install -y python3.6 python3-pip 

# Install pythons tools you need
RUN python3.6 -m pip install --upgrade setuptools pip distlib

# Install tensorflow and Keras
RUN python3.6 -m pip install tensorflow==2.6.0 keras==2.6.0 scipy==1.5.4

# Run your python file
CMD ["python3.6", "main.py"]
```

In your [`main.py`](http://main.py) file, you should have those 2 lines that will return GPU:0 if it works:

```python
import tensorflow as tf
print("GPU available: ", tf.config.list_physical_devices('GPU'))
```

Then run the Docker commands:

```bash
# Build the image
docker build -t tf_test_gpu:latest .
# Run the container to see the output
docker run --gpus all -t tf_test_gpu:latest
```

## Run a jupyter notebook

If you want to run a jupyter notebook, create this Dockerfile:

```docker
FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
# Copy files
RUN mkdir /app
WORKDIR /app
COPY . /app

# Update the system
RUN apt update
RUN apt upgrade -y

# Install python 3.6
RUN DEBIAN_FRONTEND="noninteractive" apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND="noninteractive" apt install -y python3.6 python3-pip 

# Install python tools
RUN python3.6 -m pip install --upgrade setuptools pip distlib

# Install requirements
RUN python3.6 -m pip install -r requirements.txt
RUN python3.6 -m pip install notebook

# Run your notebook in the local network on port 8888
CMD ["python3.6", "-m", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
```

With the requirements:

```python
tensorflow==2.6.0
keras==2.6.0
Pillow==8.3.2
notebook==6.4.4
matplotlib==3.3.4
scipy==1.5.4
```

Then run the Docker commands:

```bash
# Build the image
docker build -t notebook_tf_gpu:latest .
# Run the container with a volume that link your notebook to your local folder
docker run -v ${PWD}:/app -p 8888:8888 --gpus all -it notebook_tf_gpu:latest
```

Then open the notebook in your browser at [http://localhost:8888/](http://localhost:8888/).

## GitHub repo with example

## Resources

- [Nvidia Docker images](https://hub.docker.com/r/nvidia/cuda/tags?page=1&ordering=last_updated&name=11.2)
- [Tensorflow compatibility table](https://www.tensorflow.org/install/source#gpu)
- [Tensorflow Docker documentation](https://www.tensorflow.org/install/docker)
- [How to run a notebook through Docker](https://u.group/thinking/how-to-put-jupyter-notebooks-in-a-dockerfile/)
- [Docker with CUDA](https://www.celantur.com/blog/run-cuda-in-docker-on-linux/)