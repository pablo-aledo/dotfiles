FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt update && apt install -y \
    git \
    python3.10 \
    python3-pip \
    python3-venv \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel

RUN git clone https://github.com/ace-step/ACE-Step.git /app

WORKDIR /app

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install gradio>=5.0.0

RUN pip install -e .

RUN sed -i 's/server_name=server_name/server_name="0.0.0.0"/' /app/acestep/gui.py && \
    find /app -name "*.py" -exec sed -i -E 's/,? *show_download_button *= *(True|False),?//g' {} +

EXPOSE 7865

CMD ["acestep", "--port", "7865"]
