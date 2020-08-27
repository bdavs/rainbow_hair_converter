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

description = "This bot can turn your waifu rainbow. type $help to get started"

bot = commands.Bot(command_prefix=commands.when_mentioned_or('$'), description=description, case_insensitive=True)
# bot = commands.Bot(command_prefix='$')


@bot.command()
async def test(ctx, *, args):
    await ctx.send(args[-1])

def process_other_args(args):
    for arg in args:
        if arg == 'gradient':
            return True
        elif arg == 'solid':
            return False
        else:
            return False

@bot.command()
async def url(ctx, image_url:str=None, options:bool=False):
    # print(args)
    # image_url = args[-1]
    if not image_url:
        await ctx.send('No url provided')
        return
    # if len(args) > 1:
        # options = process_other_args(args)
    # else:
        # options = False
    await send_image(ctx, image_url, options)

@bot.command()
async def pic(ctx, *, args):
    try:
        attachment_string = ctx.message.attachments[0]
        x = re.search("url=\'(.+)\'", str(attachment_string))
        image_url = x.groups()[0]
        await send_image(ctx, image_url)
    except IndexError:
        await ctx.send('No image provided')

async def send_image(ctx, image_url, options):
    bot_message = await ctx.send('Beep. Boop. Processing your image...')
    try:
        await get_src_image(ctx,image_url)
        make_gif.main(single_file=INPUT_FILE,options=options)
            # url=image_url)
        await ctx.send(file=discord.File(IMAGE_FILE))
    except aiohttp.client_exceptions.InvalidURL:
        await ctx.send('Couldn\'t get that url')
        print(f"url error {image_url}\n {options}")
    except:
        await ctx.send('Uh oh, something went wrong. go yell at @bdavs')
        print("Unexpected error:", sys.exc_info()[0])
        raise
    finally:
        await bot_message.delete()



async def get_src_image(ctx,my_url):
    async with aiohttp.ClientSession() as session:
        async with session.get(my_url) as resp:
            if resp.status != 200:
                return await ctx.send('Could not download file...')
            data = io.BytesIO(await resp.read())
            # io.FileIO(INPUT_FILE,'w').write(data)
            with open(INPUT_FILE,'wb') as f:
                f.write(data.getbuffer())
            # return data
            # await channel.send(file=discord.File(data, 'cool_image.png'))

  
@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Game(name='Dreaming of polychromatic sheep | $help'))
    print(f'{bot.user.name} has connected to Discord!')

bot.run(TOKEN)