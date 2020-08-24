from tokenfile import TOKEN
from discord.ext import commands
import discord
import re
import sys
sys.path.append('/face-seg')
import convert_image

bot = commands.Bot(command_prefix='$')

@bot.command()
async def test(ctx, arg):
    await ctx.send(arg)

@bot.command()
async def url(ctx, image_url):
    convert_image.main(url=image_url)
    await ctx.send(file=discord.File('/project/output.gif'))

@bot.command()
async def pic(ctx, arg=None):
    try:
        attachment_string = ctx.message.attachments[0]
        x = re.search("url=\'(.+)\'", str(attachment_string))
        image_url = x.groups()[0]
        convert_image.main(url=image_url)
        await ctx.send(file=discord.File('/project/output.gif'))
    except IndexError:
        await ctx.send('No image provided')

  
@bot.event
async def on_ready():
    print(f'{bot.user.name} has connected to Discord!')

bot.run(TOKEN)