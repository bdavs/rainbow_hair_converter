import asyncpraw
import time
import datetime
import discord
# from secret_reddit_info import reddit
import queue
reddit = asyncpraw.Reddit()

q = queue.Queue()
used_list = []

maxAge = 3600 #(1hr) oldest comment age in seconds to prevent too many comments from posting


async def put_requests():
    reqpostSTR = "l8lpko" #monitored post      
    submission = await reddit.submission(reqpostSTR)
    comments = await submission.comments()
    await comments.replace_more(limit=None) #get all comments
    all_comments = await comments.list()

    for comment in all_comments:
        if((comment.parent_id=="t3_" + reqpostSTR) and (maxAge > time.time() - comment.created_utc)):
            if(comment.id not in used_list):

                #add to list of used comment to avoid double sending
                used_list.append(comment.id)

                ts = datetime.datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S %Z')
                em = discord.Embed(colour=0x0000AA)
                await comment.author.load() #load redditor 
                em.set_author(name=str(comment.author), icon_url=str(comment.author.icon_img))

                em.add_field(name='Request:', value=comment.body, inline=False)
                em.add_field(name='\u200b', value="[Comment Link](http://reddit.com{})".format(comment.permalink), inline=False)
                em.set_footer(text="posted at {} ".format(ts))

                # add to queue
                q.put(em)

async def get_req():
    #pop off queue (thread safe)
    try:
        item = q.get(block=False)
        q.task_done()
    except queue.Empty:
        item = None
    return(item)



if __name__ == "__main__":
    # get_req()
    put_requests()
