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
import requests
import json
#import whois
import dns.resolver, dns.rdatatype
from cryptography import x509
import socket
import ssl
import sys
import numpy as np

#custom lib
import featurePrepare as fp
import ml_loader as mll
#

load_dotenv()

dctoken = open(Path.cwd().joinpath('dc-token.txt'),'r').readline()
vttoken = open(Path.cwd().joinpath('vt-token.txt'),'r').readline()
TOKEN = os.getenv(dctoken)

description = '''URLs Scanner Bot'''
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
        vtotal = virustotal_python.Virustotal(vttoken)
        resp = vtotal.request("urls", data={"url": link}, method="POST")
        statusCde=resp.status_code
        url_id = urlsafe_b64encode(link.encode()).decode().strip("=")
        report = vtotal.request(f"urls/{url_id}")
        # pprint(report.data)
        # with open(Path(__file__).with_name('link_log.txt'), 'w') as f:
        #     f.write(str(report.data))

        if('attributes' in report.data):
            if('last_analysis_stats' in report.data['attributes']):
                await ctx.send("virustotal result : {:.2f}".format(report.data['attributes']['last_analysis_stats']
                ['malicious']) + "% possibility is malicious")
            else:
                await ctx.send("no last analysis stats")
        else:
            await ctx.send("fail get analysis stats from virustotal")
    except ValueError as ve:
        print(ve)
        await ctx.send("debug fail")

@bot.command(name='aichk')
async def _aichk(ctx, link: str):
    """aichk : link; use ai to check link"""

    try:
        features = fp.getUrlFeatures(link)
        if features is None:
            await ctx.send('Could not provide enough features to predict')
        else:
            c45_pred = mll.c45_predict(features)*100
            #ctree_pred = mll.ctree_predict(features)*100
            rf_pred=np.clip(mll.RF_predict(features), 0.1, 1)*100
            SVM_pred=np.clip(mll.SVM_predict(features), 0.3, 0.9)*100
            #MLPClamp=np.clip(mll.MLP_predict(features), 0.1, 1)
            #MLP_pred = MLPClamp*100
            #MLP_pred = mll.MLP_predict(features)*100
            await ctx.send(
              "\nC4.5 result : {:.2f}".format(c45_pred[0]) +"% possibility is malicious"
            + "\nRF result : {:.2f}".format(rf_pred[0])+"% possibility is malicious"
            + "\nSVM result : {:.2f}".format(SVM_pred[0]) +"% possibility is malicious"
            )
            #+ "\nMLP result : {:.2f}".format(MLP_pred[0]) +"% possibility is malicious")
            #"Ctree result : {:.2f}".format(ctree_pred[0])+"% possibility is malicious"
 
    except ValueError as ve:
        print(ve)
        await ctx.send("debug fail")
bot.run(dctoken)