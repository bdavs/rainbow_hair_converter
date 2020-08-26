from tokenfile import TOKEN
from discord.ext import commands
import io
import aiohttp
import discord
import re
import make_gif
import logging
import sys

#logging setup
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

IMAGE_FILE = '/project/data/output/output.gif'
INPUT_FILE = 'data/downloads/input.png'

bot = commands.Bot(command_prefix='$')

@bot.command()
async def test(ctx, arg):
    await ctx.send(arg)

@bot.command()
async def url(ctx, image_url):
    if not image_url:
        await ctx.send('No url provided')
    await send_image(ctx, image_url)
    # make_gif.main(url=image_url)
    # await ctx.send(file=discord.File(IMAGE_FILE))

@bot.command()
async def pic(ctx, arg=None):
    try:
        attachment_string = ctx.message.attachments[0]
        x = re.search("url=\'(.+)\'", str(attachment_string))
        image_url = x.groups()[0]
        await send_image(ctx, image_url)
    except IndexError:
        await ctx.send('No image provided')

async def send_image(ctx, image_url):
    bot_message = await ctx.send('Beep. Boop. Processing your image...')
    try:
        # await get_src_image(image_url)
        make_gif.main(url=image_url)
        # single_file=INPUT_FILE)
        await ctx.send(file=discord.File(IMAGE_FILE))
    except aiohttp.client_exceptions.InvalidURL:
        await ctx.send('Couldn\'t get that url')

        print("url error")
    except:
        await ctx.send('Uh oh, something went wrong. go yell at @bdavs')
        print("Unexpected error:", sys.exc_info()[0])
        raise
    finally:
        await bot_message.delete()
  
@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')


async def get_src_image(my_url):
    async with aiohttp.ClientSession() as session:
        async with session.get(my_url) as resp:
            if resp.status != 200:
                return await channel.send('Could not download file...')
            data = io.BytesIO(await resp.read())
            with open(INPUT_FILE,'wb') as f:
                f.write(data)
            # return data
            # await channel.send(file=discord.File(data, 'cool_image.png'))

bot.run(TOKEN)