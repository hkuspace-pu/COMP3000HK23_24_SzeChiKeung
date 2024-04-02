import os
from pathlib import Path
import time
from datetime import datetime
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

import dns.resolver, dns.rdatatype
from cryptography import x509
import socket
import ssl
import sys
import networkx as nx

def urlScanHdr():
    apiKey = '0229d46d-22ec-40ca-b6c7-c0cce0e6219b'
    headers = {'API-Key':apiKey, 'Content-Type':'application/json'}
    return headers

def get_urlScan_apiLink(url):
    headers = urlScanHdr()
    data={'url':url, 'visibility':'public'}
    response = requests.post('https://urlscan.io/api/v1/scan',headers=headers, data=json.dumps(data))
    apiLink = response.json()['api']
    return apiLink

def urlScan(url):
    api_response = requests.get(url, headers=urlScanHdr())
    return api_response

def getJsonFromUrlScan(urlpath):
    apiLink = get_urlScan_apiLink(urlpath)
    #time.sleep(8)
    apiJson = urlScan(apiLink)
    apiJson.json()
    return apiJson

getJsonFromUrlScan('google.com')