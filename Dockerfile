# ══════════════════════════════════════════════════════════════════════════════
# Этап 1: сборщик
# nvidia/cuda:12.1.1-devel — дает nvcc + cuBLAS заголовки для llama-cpp-python
# ══════════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# системные зависимости для сборки
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    build-essential \
    ninja-build \
    git \
    libsndfile1-dev \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# CMake 3.28 — Ubuntu 22.04 идет с 3.22 который не может разрешить CUDA arch имена
RUN wget -qO /tmp/cmake.sh \
    https://github.com/Kitware/CMake/releases/download/v3.28.6/cmake-3.28.6-linux-x86_64.sh \
 && bash /tmp/cmake.sh --skip-license --prefix=/usr/local \
 && rm /tmp/cmake.sh

# виртуальная среда в /opt/venv — предсказуемый путь для COPY в runtime этапе
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PIP_DEFAULT_TIMEOUT=300

# PyTorch (CUDA 12.1)
RUN pip install --no-cache-dir \
    torch==2.4.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# фиксируем numpy<2 ДО того как llama-cpp-python потянет numpy 2.x как зависимость
RUN pip install --no-cache-dir "numpy>=1.24.0,<2.0"

# llama-cpp-python — компилируется с CUDA/cuBLAS
# libcuda.so.1 это driver stub — не присутствует в devel образе во время сборки
# делаем симлинк на stub из toolkit чтобы линкер мог разрешить CUDA driver символы
# во время выполнения настоящий libcuda.so.1 инжектится nvidia-container-runtime
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so \
           /usr/local/cuda/lib64/stubs/libcuda.so.1 \
 && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/cuda-stubs.conf \
 && ldconfig

# GGML_CUDA=on           → GPU разгрузка через cuBLAS
# GGML_CUDA_FORCE_MMQ=on → быстрее на потребительских GPU (Turing/Ampere)
# SM список: 61=GTX10xx(Pascal) | 70=V100 | 75=RTX20xx/T4 | 80=A100 | 86=RTX30xx | 89=RTX40xx
ENV CMAKE_ARGS="-DGGML_CUDA=on -DGGML_CUDA_FORCE_MMQ=on -DCMAKE_CUDA_ARCHITECTURES=61;70;75;80;86;89"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir "llama-cpp-python==0.3.4"

# убираем stub симлинк — runtime использует настоящий драйвер с хоста
RUN rm -f /usr/local/cuda/lib64/stubs/libcuda.so.1

# остальные Python зависимости
# numpy уже зафиксирован выше — просто ставим requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# huggingface_hub + pyyaml (используется entrypoint для скачивания моделей)
RUN pip install --no-cache-dir "huggingface_hub[cli]>=0.22" pyyaml


# ══════════════════════════════════════════════════════════════════════════════
# Этап 2: выполнение
# nvidia/cuda:12.1.1-runtime — меньше, но все еще содержит cuBLAS .so нужные во время выполнения
# ══════════════════════════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# системные зависимости для выполнения
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# копируем всю виртуальную среду из сборщика
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# исходники приложения
COPY app.py app_telephony.py llm.py ./
COPY static/ ./static/
COPY config.yml ./

# точка входа
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# веса Resemblyzer (17 MB, встроенные — скачивание не нужно)
COPY models/resemblyzer/ /models/resemblyzer/

# папки + пользователь не-root
ENV HF_HOME=/cache/huggingface
ENV TRANSFORMERS_CACHE=/cache/huggingface

RUN useradd -m -u 1000 appuser \
 && mkdir -p /models /cache/huggingface \
 && chown -R appuser:appuser /app /models /cache /entrypoint.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=25)"

ENTRYPOINT ["/entrypoint.sh"]
