# Rainbow Waifu
Uses Machine Learning techniques to rainbowify your favorite anime waifu.

## Overview
Follows the basic process:
1. (optional) Downloads and saves the target image 
2. Performs Semantic Segmentation to identify key face features, especially the hair
2. Creates a mask over the hair region
3. Creates several images with different colors mixed into the masked area
4. Combines the images into a single gif

## Setup

The suggested way to run the program is using Docker as described [below](#Docker) as this ensures the system has all of the dependencies needed including pytorch


This project was built on python 3.7. To install all of the required python dependencies, run the following:
```
pip install -r requirements.txt
```

## Running
Can be ran directly from the command line 
```bash
python convert_image.py -u http://www.example.com/yourwaifuimage.jpg
```
it can take the following flags:
```bash
  -h, --help            show this help message and exit
  -f SINGLE_FILE, --single-file SINGLE_FILE
                        specify the path to a single file to run on
  -u URL, --url URL     specify a url to download from
  -o OUTPUT_FOLDER, --output_folder OUTPUT_FOLDER
                        specify the path to the folder to write to (default: .)
```     
#### note: only use either a single file or a url, not both                   


## Discord bot

Includes a script to run a discord bot using discord<span></span>.py. 
It requires you to set up your own bot on discord beforehand [as described here](https://discordpy.readthedocs.io/en/latest/discord.html) 

Once you have the bot's token, open the file `tokenfile.py` and enter your token in  the quotes 
```python
TOKEN = 'placeyourtokenhere'
```

run the bot with the following:
```python
python discord_bot.py
```

Add your bot to your server and you can now use it.

The easiest way is to supply it with the URL of the image. 
In a channel that the bot has joined, simply type: `$url http://www.example.com/yourwaifuimage.jpg` and the bot should respond a few seconds later with the image.

The other way is to upload an image and in the comment field when uploading, add the text `$pic` then send

## Docker
Included is a Dockerfile to build the whole system with its needed dependencies

```bash
docker build --tag rainbow_waifu .
```

## Results
![alt text](data/samples/anime_hair_example.png "Input")
![alt text](data/samples/anime_hair_example_output.gif "Output")