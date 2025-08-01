FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install flask ultralytics pillow numpy opencv-python

EXPOSE 8080

CMD ["python", "app.py"]
