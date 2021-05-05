## Install requirements
I have tested the code on Python 3.6.9
```shell
pip install -r requirements.txt
```

## Directory structure
Create and place the csv files in data directory.

Create a features directory

## Train the model.
run the script 
```shell
ts.py
```

## Run API flask server
```shell
python api.py
```

Test it by providing values as json. You need to provide only `cp`,`kvFst`,`kvTrgger`,`ripple_angle` and `wave_angle` values.
```shell
curl -X 'POST' 'http://127.0.0.1:5000/api/wave_type' -H 'Content-Type: application/json' -d '{"row": "1111,-1.81898940354586E-12,-2.59855629077979E-13,41.1859251657097,41.1859251657097"}'
```

## Docker Setup
To run it on host install 
```shell
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list |\
    sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
```

checkout this on docker hub
for cuda 10.2
```text
nvidia/cuda:10.2-cudnn7-devel
```

for cuda 11.2
```text
nvidia/cuda:11.2.0-cudnn8-devel
```

Make sure nvidia-smi is working using
```shell
docker run --gpus all nvidia/cuda:11.2.0-cudnn8-devel nvidia-smi
```
To run API in docker, you need to build it by following command and run it.
```shell
docker build -t wave_type -f Dockerfile .
```
Run docker flask app
```shell
docker run -d -p 5000:9007 wave_type
```
