FROM python:3.11.14
WORKDIR /app
RUN mkdir Data
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*
COPY . .

RUN pip install --default-timeout=3000 -r requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1
EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]



