FROM ubuntu:bionic as full-base-torch

ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /app

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && apt-get update \
    && apt-get install -y git software-properties-common python3-pip python3-opencv cifs-utils \
    && apt-get -y autoremove \
    && python3 -m pip install --upgrade pip \
#     && python3 -m pip install torch==1.10.0 torchvision==0.11.1 torchaudio==0.10.0
   && pip3 install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

FROM full-base-torch as builder

RUN --mount=type=cache,target=/root/.cache \
    pip install poetry \
    && poetry config virtualenvs.create false
COPY poetry.lock pyproject.toml README.md /app/
RUN --mount=type=cache,target=/root/.cache \
    python3 -m venv $VIRTUAL_ENV \
    && pip install --upgrade pip \
    && poetry install --no-root --only main

FROM full-base-torch as yolov7

COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV

RUN mkdir -p /app/config \
    && mkdir -p /app/weights

COPY . /app

COPY src/ /app/
COPY config.yaml /app/config.yaml
COPY entrypoint.sh /app/entrypoint.sh
COPY data/hyp.scratch.custom.yaml /app/config/hyp.scratch.yaml
COPY weights/ /app/weights/

RUN python3 -m pip install -r requirements.txt --no-cache-dir

RUN mkdir -p /data/weights && mkdir /data/config
#    && echo "python3 app.py" >> ./entrypoint.sh \
#    && echo "python3 -m flask run -p 5003 --host=0.0.0.0" >> ./entrypoint.sh \
# && echo "python3 train.py --device \$ENV_DEVICE --img 512 --batch \$ENV_BATCH_SIZE --epochs \$ENV_EPOCHS --data /data/config/dataset.yaml --hyp /data/config/hyp.scratch.yaml --weights /data/weights/best.pt" >> ./entrypoint.sh \

EXPOSE 5003

ENTRYPOINT ["/bin/bash", "/app/entrypoint.sh"]

ARG BUILD_DATE
ARG REF_NAME
ARG VERSION

LABEL org.opencontainers.image.authors="Benedict Lindner <benedict.lindner@synthavo.de>"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.ref.name="${REF_NAME}"
LABEL org.opencontainers.image.version="${VERSION}"
