FROM python:3.11.14
WORKDIR /app
RUN mkdir Data
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*
COPY . .

RUN pip install -r requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1
EXPOSE 5000
CMD ["python","app.py"]



