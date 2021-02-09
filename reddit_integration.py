import praw
import time
import datetime
import discord
from secret_reddit_info import reddit
import queue

q = queue.Queue()
used_list = []

subreddit = reddit.subreddit("Rainbow_Waifus") 
reqpostSTR = "l8lpko" #monitored post              # jvkj0j l8lpko
reqpost = reddit.submission(reqpostSTR)
maxAge = 3600 #oldest comment age in seconds to prevent 


def put_requests():
    all_comments = reqpost.comments.list()
    for comment in all_comments:
        if((comment.parent_id=="t3_" + reqpostSTR) and (maxAge > time.time() - comment.created_utc)):
            if(comment.id not in used_list):
                used_list.append(comment.id)

                ts = datetime.datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S %Z')

                em = discord.Embed(colour=0x0000AA)
                em.set_author(name=str(comment.author), icon_url=str(comment.author.icon_img))
                em.add_field(name='Request:',
                            value=comment.body, inline=False)
                em.add_field(name='\u200b', value="[Comment Link](http://reddit.com{})".format(comment.permalink))
                em.set_footer(text="posted at {} ".format(ts))
                            # icon_url="https://cdn.discordapp.com/emojis/554730061463289857.gif")

                # text = ""
                # text += "Request by: u/" + str(comment.author) +"\n"
                # text += "comment link: www.reddit.com/comments/" + reqpostSTR + "/Requests/" + str(comment) +"\n"
                # text += str(comment.body) +"\n"
                # text += "-----------\n"
                # return(em)
                # await ctx.send(embed=em)
                # print("adding comment to queue")
                q.put(em)

def get_req():
    try:
        item = q.get(block=False)
        q.task_done()
    except queue.Empty:
        item = None
    return(item)



if __name__ == "__main__":
    # get_req()
    put_requests()
