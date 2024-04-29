import pandas as pd
from virustotal_python import Virustotal
from base64 import urlsafe_b64encode
from pathlib import Path

class VirusTotalScanner:
    def __init__(self, vt_token):
        self.virustotal = Virustotal(API_KEY=vt_token)

    def scan_url(self, url):
        url_id = urlsafe_b64encode(url.encode()).decode().strip("=")
        response = self.virustotal.request(f"urls/{url_id}")
        return response

    def process_urls(self, input_csv, output_csv):
        df = pd.read_csv("Malicious.csv", header=None)
        df.columns = ['URL']
        results = {'URL': [], 'Malicious Status': [], 'Total Votes': []}

        for url in df['URL']:
            try:
                scan_result = self.scan_url(url)
                malicious_votes = scan_result.data['attributes']['last_analysis_stats']['malicious']
                total_votes = sum(scan_result.data['attributes']['last_analysis_stats'].values())
                results['URL'].append(url)
                results['Malicious Status'].append(malicious_votes)
                results['Total Votes'].append(total_votes)
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                results['URL'].append(url)
                results['Malicious Status'].append('Error')
                results['Total Votes'].append('Error')

        results_df = pd.DataFrame(results)
        results_df.to_csv(r'C:\Users\sauls\source\repos\COMP3000HK23_24_SzeChiKeung\Results.csv', index=False)

if __name__ == "__main__":
    vt_token = open(Path.cwd().joinpath('vt-token.txt'), 'r').readline().strip()
    scanner = VirusTotalScanner(vt_token)
    scanner.process_urls("input_urls.csv", "virustotal_results.csv")
