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

def getUrlFeatures(url):
    apiJson = getJsonFromUrlScan(url)
    if apiJson == None:
        return None
    else:
        with open('link_log.txt', 'w') as f:
            json.dump(apiJson, f)
        #apiJson
        timing = 0
        pageRank = 0
        if ('redirectResponse' in apiJson['data']['requests'][0]['request']):
            if ('timing' in apiJson['data']['requests'][0]['request']['redirectResponse']):
                timing = apiJson['data']['requests'][0]['request']['redirectResponse']['timing']
                pageRank = time.localtime(timing['connectEnd'] - timing['connectStart']).tm_sec
        domains = len(apiJson['lists']['domains'])#['data']
        ips = len(apiJson['lists']['ips'])
        redirect = len(apiJson['data']['requests'])

        data = { 'url_too_long':[longUrl(url),],
              'url_contain_@':[containAtSymbol(url),],
              'url_contain_hyphen' :[containHyphenSymbol(url),],
              'redirects'  :[ redirect,],
              'not_indexed_by_google'   :[googleIndexed(url),],
              'certificate_age'    :[-getCertExpDate(url),],
              'TTL'     :[chkredirect(url),],
              'ip_address_count'      :[ips,],
              'count_domain_occurrences'       :[domains,],
              'page_rank_decimal'        :[pageRank], }

        df = pd.DataFrame(data)
        # df.index = ['url_too_long',	'url_contain_@','url_contain_hyphen	redirects',	'not_indexed_by_google','certificate_age', 'TTL', 'ip_address_count', 'count_domain_occurrences', 'page_rank_decimal']
        # df.columns = [longUrl(url),
        #         containAtSymbol(url),
        #         containHyphenSymbol(url),
        #         redirect,
        #         googleIndexed(url),
        #         getCertExpDate(url),
        #         chkredirect(url),
        #         ips,
        #         domains,
        #         pageRank]
        return df

def longUrl(url):
    urllen = len(url)
    if urllen <= 54:
        return 0
    elif urllen >= 54 and urllen <= 75:
        return 0.5
    else:
        return 1

def containAtSymbol(url):
    if '@' in url:
        return 1
    else:
        return 0

def containHyphenSymbol(url):
    if '-' in url:
        return 1
    else:
        return 0

def chkredirect(url):
    try:
        aws = dns.resolver.resolve(url)
        TTL = aws.rrset.ttl
    except:
        return 0
    return TTL
    
def getCertExpDate(url):
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    expdays = 0
    try:
        sock = socket.create_connection((url, 443))
        ssock = context.wrap_socket(sock, server_hostname=url)
        # get cert in DER format
        data = ssock.getpeercert(True)
        pem_data = ssl.DER_cert_to_PEM_cert(data)
        cert_data = x509.load_pem_x509_certificate(str.encode(pem_data))
        #print("Expiry date:", cert_data.not_valid_after)
        expdate = cert_data.not_valid_after_utc
        expdays = expdate.replace(tzinfo=None)
        expdays = (expdays - datetime.now()).days
        #print(expdays)
    except:
        return 0
    return expdays

def googleIndexed(url):
    queryHdr = 'site:'
    query = ''
    if 'http' in url:
        queryHdr = queryHdr+url.split('/')[2]
    else:
        query = queryHdr+url
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.37 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36'}
    url = 'https://www.google.com/search?q='+query
    res = requests.get(url, headers)
    if not res:
        return 0
    else:
        return 1

def urlScanHdr():
    apiKey = '0229d46d-22ec-40ca-b6c7-c0cce0e6219b'
    headers = {'API-Key':apiKey, 'Content-Type':'application/json'}
    return headers

def get_urlScan_apiLink(url):
    headers = urlScanHdr()
    data={'url':url, 'visibility':'public'}
    response = requests.post('https://urlscan.io/api/v1/scan',headers=headers, data=json.dumps(data))
    if 'api' in response.json():
        apiLink = response.json()['api']
    else:
        apiLink = "";
    return apiLink

def urlScan(url):
    api_response = requests.get(url, headers=urlScanHdr())
    return api_response

def getJsonFromUrlScan(urlpath):
    apiLink = get_urlScan_apiLink(urlpath)
    if(apiLink == ""):
        return None
    start = time.time()
    while(True):
        apiJson = urlScan(apiLink).json()
        end = time.time()
        if 'message' in apiJson and 'status' in apiJson:
            if apiJson['message'] == 'Scan is not finished yet' or apiJson['status'] == 404:
                continue
            elif apiJson['message'] == '':
                break 
        else:
            return apiJson
        
        if time.localtime(end-start).tm_sec > 15:
            break

    return None