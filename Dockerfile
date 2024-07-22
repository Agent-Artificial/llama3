FROM nvcr.io/nvidia/tensorrt:24.06-py3

SHELL ["/bin/bash", "-c"]

COPY ./ /app


RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y git

RUN pip install -r /app/requirements.txt

ENV HF_TOKEN $HF_TOKEN

CMD ["python", "/app/api.py"]
