FROM python:slim
#FROM pytorch/pytorch

WORKDIR /project

RUN apt update && apt install -y \
    libgtk2.0-dev \
    libgl1-mesa-glx \
     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "./discord_bot.py" ]