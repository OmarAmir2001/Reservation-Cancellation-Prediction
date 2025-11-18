
FROM python:3.9-slim


WORKDIR /app


RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


RUN mkdir -p data models


COPY train.py .
COPY predict.py .
COPY data/ data/


RUN python train.py


RUN useradd --create-home --shell /bin/bash appuser
USER appuser


EXPOSE 5000


CMD ["python", "predict.py"]