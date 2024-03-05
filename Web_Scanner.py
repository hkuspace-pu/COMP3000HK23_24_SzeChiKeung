import os
from pathlib import Path
#discord
import discord
from discord.ext import commands
from dotenv import load_dotenv
#virustotal
import virustotal_python
from pprint import pprint
from base64 import urlsafe_b64encode
import pandas as pd

load_dotenv()

dctoken = open(Path.cwd().joinpath('dc-token.txt'),'r').readline()
vttoken = open(Path.cwd().joinpath('vt-token.txt'),'r').readline()
TOKEN = os.getenv(dctoken)

description = '''Web Scaner Bot'''
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix='?', description=description, intents=intents)
@bot.event
async def on_ready():
    print(f'Logged in as {bot.user} (ID: {bot.user.id})')
    print('------')

@bot.command(name='ping')
async def _ping(ctx):
    """ping the bot"""
    await ctx.send('pong!')

@bot.command(name='chklink')
async def _chklink(ctx, link: str):
    """chklink : link; for check link status"""
    statusCde = "ok"
    try:
        with virustotal_python.Virustotal(vttoken) as vtotal:
            resp = vtotal.request("urls", data={"url": link}, method="POST")
            statusCde=resp.status_code
            url_id = urlsafe_b64encode(link.encode()).decode().strip("=")
            report = vtotal.request(f"urls/{url_id}")
            pprint(report.data)
            with open(Path(__file__).with_name('link_log.txt'), 'w') as f:
                f.write(str(report.data))
    except:
        await ctx.send("debug fail")

bot.run(dctoken)

##vt api