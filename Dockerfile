FROM python:3.11-bookworm
WORKDIR /app
COPY . .
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install -r requirements.txt

# ENV DISPLAY=host.docker.internal:0.0

# CMD ["/usr/local/bin/python3", "runner.py"]
