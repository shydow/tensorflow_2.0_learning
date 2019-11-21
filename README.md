# tensorflow_2.0_learning

## Env Setup

1. setup virtual env(optional)
    ```shell
    virtualenv venv
    source venv/bin/activate
    ```
1. install dependencies
    ```shell
    pip install -r requirements.txt
    ```
1. open project in vscode
    ```shell
    code .
    ```


## docker running config 
```shell
docker run --runtime=nvidia -it -p 80:8888 --rm tensorflow/tensorflow:latest-gpu-py3-jupyter
```
