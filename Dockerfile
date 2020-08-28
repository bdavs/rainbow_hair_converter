FROM python:3.6-slim
#FROM pytorch/pytorch

WORKDIR /project

RUN apt update && apt install -y --no-install-recommends\
    libgtk2.0-dev \
    libgl1-mesa-glx \
     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./discord_bot.py" ]