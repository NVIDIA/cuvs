FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir requests boto3 numpy
COPY run.py .
CMD ["python", "-u", "run.py"]
