# Использование официального образа PyTorch с поддержкой CUDA
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# Установка переменных окружения для неинтерактивной установки
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Установка системных зависимостей для OpenCV, vLLM и других библиотек
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Обновление pip
RUN pip install --upgrade pip

# Копирование файла зависимостей
COPY requirements.txt .

# Установка зависимостей
# Примечание: vLLM скачает готовые бинарники для Linux внутри контейнера
RUN pip install --no-cache-dir -r requirements.txt

# Копирование кода проекта
COPY . .

# Команда для запуска (можно переопределить в docker-compose)
CMD ["python", "vlm_pipeline.py"]
