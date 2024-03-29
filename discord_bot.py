try:
    # get discord token from file
    from tokenfile import DISCORD_TOKEN
except ModuleNotFoundError:
    #get discord token from environment
    try:
        import os
        DISCORD_TOKEN = os.environ['DISCORD_TOKEN']
    except KeyError:
        print("could not get token")
from discord.ext import commands,tasks
import random as rand
import io
import aiohttp
import asyncio
import threading
import discord
import re
import logging
import sys

import make_gif
import reddit_integration

#logging setup
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

IMAGE_FILE = '/project/data/output/output.gif'
INPUT_FILE = 'data/downloads/input.png'

description =   """This bot can turn your waifu rainbow. 
                Take a look at '$help url' and '$help pic' to get started  
                quick example:
                $url f3 gradient https://www.kindpng.com/picc/m/274-2748314_freetoedit-menherachan-animegirl-animecute-png-kawaii-anime-girl.png

                Works best on clean images where the hair is easily defined and the face is easily visible
                Prefix for all commands is $
                """

bot = commands.Bot(command_prefix=commands.when_mentioned_or('$'), description=description, case_insensitive=True)
# bot = commands.Bot(command_prefix='$')


@bot.command(description="get a random waifu")
async def random(ctx, *args):
    """Rainbows a random waifu off thiswaifudoesnotexist.net *cursed*
    WARNING: I have no control over what abominations come from this site
    Milage may vary
    
    when you get an image if you want to keep messing with it, use this url:
    https://www.thiswaifudoesnotexist.net/example-{randNum}.jpg
    and replace {randNum} with the number provided and use the $url command as normal

    All normal options apply here
    """
    randNum = rand.randint(1,50000)
    randWaifu = f"https://www.thiswaifudoesnotexist.net/example-{randNum}.jpg"
    if len(args)>0:
        # newargs = ' '.join(args) + ' ' +  randWaifu
        # newargs = list(args).append(randWaifu)
        newargs = args + (randWaifu,)
    else:
        newargs = randWaifu
    await url(ctx,*args,randWaifu)
    await ctx.send(f"random waifu number {randNum} from https://www.thiswaifudoesnotexist.net\n{randWaifu}")


channel_to_join = None
@tasks.loop(seconds=5.0)
async def get_req():
    if channel_to_join is None:
        return

    await reddit_integration.put_requests()

    em = await reddit_integration.get_req()
    if em:
        await channel_to_join.send(embed=em)

@bot.command()
@commands.has_permissions(administrator=True, hidden=True)
async def requestJoin(ctx, *args):
    """joins the bot to check for request thread updates (admin only)
    """
    global channel_to_join
    channel_to_join = ctx.channel
    if not get_req.is_running():
        get_req.start()
    await ctx.send("Joined {} channel to send request thread updates".format(channel_to_join))


@bot.command()
async def test(ctx, *args,hidden=True):
    """This is just used for testing, please ignore
    seriously, it doesn't do anything cool
    here is all the code for this function:

    @bot.command()
    async def test(ctx, *args):
        if args:
            await ctx.send(' '.join(args))
        else: 
            await ctx.send("test")
    """

    if args:
        await ctx.send(' '.join(args))
    else: 
        await ctx.send("test")
    
@bot.command(aliases=['inv'])
async def invite(ctx, *args):
    """Generate an invite link to add this bot to another server
    Requires user to have Manage Server permissions on the target server
    """
    perms = discord.Permissions(permissions=1074063424)
    invite = discord.utils.oauth_url(bot.user.id,permissions=perms)
    await ctx.send(invite)

default_options={'gradient':0,'num_colors':16,'fill_level':4}
url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
def process_args(args):
    image_url=[]
    options=default_options
    for arg in args:
        if re.match(url_regex,arg) is not None:
            image_url.append(arg)
        elif arg == 'gradient':
            options['gradient']=1
        elif arg == 'solid':
            options['gradient']=0
        elif arg[0] == 'f':
            options['fill_level']=int(arg[1:])
        elif arg[0] =='c':
            options['num_colors']=int(arg[1:])
        else:
            options['gradient']=0
    # print(options)
    if len(image_url) < 1:
        image_url = None
    return options,image_url

@bot.command(aliases=['link'])
async def url(ctx, *args):
    """Get your waifu rainbowed by url
    The following are all valid args:
        gradient - apply a gradient rainbow instead of a solid one
        solid - (default) apply a solid color effect
        f# - fill level - any number from 0-9; (default: 4)
            increase this if not enough of the hair is caught
            decrease this if too much that is not hair is caught
        c# - colors used - number of colors the gif will transition between (default: 16)
            increasing this can slow down the conversion process

        examples
            $url https://www.thiswaifudoesnotexist.net/example-12345.jpg
                basic example, all defaults
            $url f7 c8 gradient https://www.thiswaifudoesnotexist.net/example-12345.jpg
                f7 - I want more hair filled
                c8 - switch between 8 colors
                gradient - I want a gradient rainbow
            $url solid c10 f3 https://www.thiswaifudoesnotexist.net/example-12345.jpg
                solid - I want a solid rainbow
                f3 - some of the face is filled, fill less
                c10 - switch between 10 colors
    """
    if len(args) == 0:
        ctx.send('No url provided')
        return

    # print(args)
    options,image_url = process_args(args)

    if image_url is None:
        await ctx.send('No url provided')
        return
    if len(args) > 1:
        await send_image(ctx, image_url, options)
    else:
        await send_image(ctx, image_url,default_options)
        
@bot.command()
async def maskurl(ctx, *args):
    """You color in what you want rainbowed, I'll do the rest (via url)
    use a pair of urls, first is the original image, second is the same image with the hair (or whatever) colored in
    The following are all valid args:
        gradient - apply a gradient rainbow instead of a solid one
        solid - (default) apply a solid color effect
        c# - colors used - number of colors the gif will transition between (default: 16)
            increasing this can slow down the conversion process

        examples
            $maskurl https://www.thiswaifudoesnotexist.net/example-12345.jpg https://www.thiswaifudoesnotexist.net/example-12345-masked-edit.jpg
                basic example, all defaults
            $maskurl c-50 gradient https://www.thiswaifudoesnotexist.net/example-12345.jpg https://www.thiswaifudoesnotexist.net/example-12345-masked-edit.jpg
                f7 - I want more hair filled
                c-50 - switch between 50 colors and reverse direction
                gradient - I want a gradient rainbow
            $maskurl c10 solid https://www.thiswaifudoesnotexist.net/example-12345.jpg https://www.thiswaifudoesnotexist.net/example-12345-masked-edit.jpg
                solid - I want a solid rainbow
                c10 - switch between 10 colors
    """
    if len(args) == 0:
        ctx.send('No url provided')
        return

    # print(args)
    options,image_url = process_args(args)

    if image_url is None:
        await ctx.send('No url provided')
        return
    if len(args) > 1:
        await send_diff_image(ctx, image_url[0], image_url[1], options)
    else:
        await send_diff_image(ctx, image_url[0], image_url[1], default_options)

@bot.command()
async def maskpic(ctx, *args):
    """You color in what you want rainbowed, I'll do the rest (via picture upload)
    To use, upload your waifu's pic  and what you want rainbowed with the comment \"$maskpic\" and any other parameters

    The following are all valid args:
        gradient - apply a gradient rainbow instead of a solid one
        solid - (default) apply a solid color effect
        c# - colors used - number of colors the gif will transition between (default: 16)
            increasing this can slow down the conversion process and lower the gif's resolution

        examples
            $pic 
                basic example, all defaults
            $pic c-50 gradient 
                c-50 - switch between 50 colors and reverse direction
                gradient - I want a gradient rainbow
            $pic solid c10
                solid - I want a solid rainbow
                c10 - switch between 10 colors
    """
    try:
        attachment_string1 = ctx.message.attachments[0]
        image_url1=attachment_string1.url
        attachment_string2 = ctx.message.attachments[1]
        image_url2 = attachment_string2.url
        if len(args) > 0:
            options,_ = process_args(args)
            await send_diff_image(ctx, image_url1, image_url2, options)
        else:
            await send_diff_image(ctx, image_url1, image_url2, default_options)
    except IndexError:
        await ctx.send('No image provided or only 1 image provided, please provide 2')
    except:
        await ctx.send('Uh oh, something went wrong. go yell at @bdavs')
        print("Unexpected error:", sys.exc_info()[0])
        raise

@bot.command()
async def pic(ctx, *args):
    """Get your waifu rainbowed by upload
    To use, upload your waifu's pic with the comment \"$pic\" and any other parameters

    The following are all valid args:
        gradient - apply a gradient rainbow instead of a solid one
        solid - (default) apply a solid color effect
        f# - fill level - any number from 0-9; (default: 4)
            increase this if not enough of the hair is caught
            decrease this if too much that is not hair is caught
        c# - colors used - number of colors the gif will transition between (default: 16)
            increasing this can slow down the conversion process and lower the gif's resolution

        examples
            $pic 
                basic example, all defaults
            $pic f7 c8 gradient 
                f7 - I want more hair filled
                c8 - switch between 8 colors
                gradient - I want a gradient rainbow
            $pic solid c10 f3 
                solid - I want a solid rainbow
                f3 - some of the face is filled, fill less
                c10 - switch between 10 colors
    """
    try:
        attachment_string = ctx.message.attachments[0]
        # x = re.search("url=\'(.+)\'", str(attachment_string))
        # image_url = x.groups()[0]
        image_url=attachment_string.url

        if len(args) > 0:
            options,_ = process_args(args)
            await send_image(ctx, image_url, options)
        else:
            await send_image(ctx, image_url,default_options)
    except IndexError:
        await ctx.send('No image provided')
    except:
        await ctx.send('Uh oh, something went wrong. go yell at @bdavs')
        print("Unexpected error:", sys.exc_info()[0])
        raise

async def send_image(ctx, image_url, options=None):
    bot_message = await ctx.send(f'Beep. Boop. Processing image for {ctx.message.author.name}...')
    try:
        # if image_url.split('.')[-1].lower() not in ['png', 'jpg', 'jpeg']:
        #     await ctx.send('Couldn\'t get that url or there was no image at that address')
        #     print(f"url error {image_url}")

        data = await get_src_image(ctx,image_url[0])
        animated_gif = make_gif.main(input_stream=data,options=options)
            # url=image_url)
        # await ctx.send(file=discord.File(IMAGE_FILE))
        await ctx.send(file=discord.File(animated_gif, 'rainbow_waifu.mp4'))
    except aiohttp.client_exceptions.InvalidURL:
        await ctx.send('Couldn\'t get that url or there was no image at that address')
        print(f"url error {image_url[0]}")
    except discord.errors.HTTPException:
        await ctx.send('Sorry, the resulting GIF was too large. Please try uploading a smaller image')
        print(f"gif too big")
    except:
        await ctx.send('Uh oh, something went wrong. Go yell at @bdavs')
        print("Unexpected error:", sys.exc_info()[0])
        raise
    finally:
        await bot_message.delete()

async def send_diff_image(ctx, image_url1, image_url2, options=None):
    bot_message = await ctx.send(f'Beep. Boop. Processing image for {ctx.message.author.name}...')
    try:
        # if image_url.split('.')[-1].lower() not in ['png', 'jpg', 'jpeg']:
        #     await ctx.send('Couldn\'t get that url or there was no image at that address')
        #     print(f"url error {image_url}")

        data1 = await get_src_image(ctx,image_url1)
        data2 = await get_src_image(ctx,image_url2)
        
        animated_gif = make_gif.main_dual(input_stream_image=data1, input_stream_mask=data2, options=options)
            # url=image_url)
        # await ctx.send(file=discord.File(IMAGE_FILE))
        await ctx.send(file=discord.File(animated_gif, 'rainbow_waifu.mp4'))
    except aiohttp.client_exceptions.InvalidURL:
        await ctx.send('Couldn\'t get that url or there was no image at that address')
        print(f"url error {image_url1} or {image_url2}")
    except discord.errors.HTTPException:
        await ctx.send('Sorry, the resulting GIF was too large. Please try uploading a smaller image')
        print(f"gif too big")
    except:
        await ctx.send('Uh oh, something went wrong. Go yell at @bdavs')
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

            # with open(INPUT_FILE,'wb') as f:
                # f.write(data.getbuffer())
            # return data.getbuffer()
            return data


  
@bot.event
async def on_ready():
    await bot.change_presence(activity=discord.Game(name='Dreaming of polychromatic sheep | $help'))
    print(f'{bot.user.name} has connected to Discord!')


bot.run(DISCORD_TOKEN)
# bot.close()