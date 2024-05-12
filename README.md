# Malicious URL Detection Bot for Discord

## Overview
This project develops a Discord bot that enhances cybersecurity by leveraging advanced machine learning (ML) algorithms to detect and analyze malicious URLs. Designed to integrate seamlessly with Discord servers, this bot provides real-time analysis of URLs, helping users identify and avoid potential cyber threats. The bot utilizes a combination of Support Vector Machine (SVM), Random Forests (RF), and C4.5 decision tree algorithms to assess the maliciousness of URLs compared against a database enriched by VirusTotal and URLscan.io.

## Features
- **Real-Time URL Analysis**: Instantly analyzes URLs submitted by users within any Discord server.
- **Advanced Machine Learning Models**: Incorporates SVM, RF, and C4.5 algorithms for accurate detection of malicious URLs.
- **Integration with VirusTotal and URLscan.io**: Leverages external APIs to enhance detection capabilities with comprehensive threat data.
- **User-Friendly Interactions**: Simple command-based interactions for users to check URLs and receive feedback on potential threats.
- **Continuous Learning and Updates**: Regularly updated ML models to adapt to new and evolving cyber threats.

## Requirements
- **Python**: >= 3.11
- **Discord account**: Create a Discord account and get the bot API key.
- **Virustotal account**: Create a Virustotal accound and get the Virustotal API key.
- **URLscan.io account**: Create a URLscan.io accound and get the URLscan.io API key.
- **Trained model**: It's too large that can not pull on the Github, download the trained model at this:
  
## Installation
**git clone https://github.com/hkuspace-pu/COMP3000HK23_24_SzeChiKeung.git**


## Usage
After installation and fulfill the requirements, you can start the bot with:

**python Web_Scanner.py**


Use the following command in Discord to check URLs through Virustotal API:

**?chklink [URL to check]**

Use the following command in Discord to check URLs through MLs:

**?aichk [URL to check]**


## Credits
* Rami M. Mohammad, Fadi Thabtah, and Lee McCluskey from the University of Huddersfield and Canadian University of Dubai for the awesome dataset [https://huggingface.co/datasets/FredZhang7/malicious-website-features-2.4M]
