FROM pytorch/pytorch

WORKDIR /project

RUN apt update && apt install -y \
    libgtk2.0-dev \
    libgl1-mesa-glx \
     && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install --no-cache-dir discord.py matplotlib opencv-python pillow 

COPY . .

CMD [ "python", "./discord_bot.py" ]