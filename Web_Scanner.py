import os
from pathlib import Path
#discord
import discord
from discord.ext import commands
from dotenv import load_dotenv
#virustotal
import virustotal_python
from pprint import pprint


import pandas as pd

load_dotenv()

dctoken = open(Path.cwd().with_name('dc-token.txt'),'r').readline()
vttoken = open(Path.cwd().with_name('vt-token.txt'),'r').readline()
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

@bot.command(name='chkdomain')
async def _chkdomain(ctx, domain: str):
    """chkdomain : domain; for check domain status"""
    with virustotal_python.Virustotal(vttoken) as vtotal:
        resp = vtotal.request(f"domains/{domain}")
        pprint(resp.data)
        with open(Path(__file__).with_name('domain_log.txt'), 'w') as f:
            f.write(str(resp.data))

    await ctx.send("debug ok")

bot.run(dctoken)

##vt api