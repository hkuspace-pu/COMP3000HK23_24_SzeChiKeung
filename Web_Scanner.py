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

#custom lib
import featurePrepare as fp
import ml_loader as mll
#

load_dotenv()

dctoken = open(Path.cwd().joinpath('dc-token.txt'),'r').readline()
vttoken = open(Path.cwd().joinpath('vt-token.txt'),'r').readline()
TOKEN = os.getenv(dctoken)

description = '''Web Scaner Bot'''
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

# def chkredirect(url):
#     aws = dns.resolver.resolve(url)
#     TTL = aws.rrset.ttl
#     return TTL
# def getCertExpDate(url):
#     context = ssl.create_default_context()
#     context.check_hostname = False
#     context.verify_mode = ssl.CERT_NONE
#     with socket.create_connection((url, 443)) as sock:
#         with context.wrap_socket(sock, server_hostname=hostname) as ssock:
#             # get cert in DER format
#             data = ssock.getpeercert(True)
#             pem_data = ssl.DER_cert_to_PEM_cert(data)
#             cert_data = x509.load_pem_x509_certificate(str.encode(pem_data))
#             #print("Expiry date:", cert_data.not_valid_after)
#             expdate = cert_data.not_valid_after
# return expdate

def urlScan(url):
    apiKey = ''
    headers = {'API-Key':apikey, 'Content-Type':'application/json'}
    data={'url':url, 'visibility':'public'}
    response = requests.post('https://urlscan.io/api/v1/scan',headers=headers, data=json.dumps(data))
    apilink = response.json()['api']
    api_response = requests.get(apiLink, headers=headers)


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

@bot.command(name='aichk')
async def _aichk(ctx, link: str):
    """aichk : link; use ai to check link"""
    try:
        features = fp.getUrlFeatures(link)
        if features is None:
            await ctx.send('Could not provide enough features to predict')
        pred = mll.ctree_predict(features)
        await ctx.send( "score :"+str(pred[0]) )
    except ValueError as ve:
        print(ve)
        await ctx.send("debug fail")

bot.run(dctoken)