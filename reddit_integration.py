import praw
import time
import datetime
import discord
from secret_reddit_info import reddit


subreddit = reddit.subreddit("Rainbow_Waifus") 
reqpostSTR = "l8lpko" #monitored post              # jvkj0j l8lpko
reqpost = reddit.submission(reqpostSTR)
maxAge = 30000000 #oldest comment age in seconds to prevent 


async def get_requests(ctx):

    for comment in subreddit.stream.comments():

        if((comment.parent_id=="t3_" + reqpostSTR) and (maxAge > time.time() - comment.created_utc)):
            ts = datetime.datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S %Z')

            em = discord.Embed(colour=0xAA0000)
            em.set_author(name=str(comment.author), icon_url=str(comment.author.icon_img))
            em.add_field(name='Request:',
                        value=comment.body, inline=False)
            em.add_field(name='Comment Link:',
                        value="http://reddit.com" + comment.permalink, inline=False)
            em.set_footer(text="posted at {} ".format(ts))
                        # icon_url="https://cdn.discordapp.com/emojis/554730061463289857.gif")

            # text = ""
            # text += "Request by: u/" + str(comment.author) +"\n"
            # text += "comment link: www.reddit.com/comments/" + reqpostSTR + "/Requests/" + str(comment) +"\n"
            # text += str(comment.body) +"\n"
            # text += "-----------\n"
            # return(em)
            await ctx.send(embed=em)


if __name__ == "__main__":
    get_requests(None)
