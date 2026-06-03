FROM continuumio/miniconda3:latest

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY train_cnn.yml dgl_env.yml mobiraph.yml ./

RUN conda env create -f train_cnn.yml
RUN conda env create -f dgl_env.yml
RUN conda env create -f mobiraph.yml

COPY . .

RUN mkdir -p /app/data /app/checkpoints /app/models /app/output

ENTRYPOINT ["python", "-u", "main.py"]