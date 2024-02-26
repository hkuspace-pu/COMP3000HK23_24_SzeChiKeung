using VirusTotalNet.Results;
using VirusTotalNet;

namespace WebSniffer
{

    public class VirusTotalUtilities
    {
        public static string _Token { get; set; } = string.Empty;
        public static void InitToken()
        {
            string tmp = string.Empty;
            using (var sr = new StreamReader(Path.Combine(Environment.CurrentDirectory, "token_vt.inf")))
            {
                tmp = sr.ReadToEnd();
            }
            _Token = tmp;
        }
        public static UrlScanResult testVTCheck(string url = "")
        {
            var result = new UrlScanResult();

            InitToken();

            if (_Token == string.Empty) return result;
            if (url == string.Empty) return result;

            VirusTotal vt = new VirusTotal(_Token);
            result = Task<UrlScanResult>.Run(() => { return vt.ScanUrlAsync(url).Result; }).Result;

            Console.WriteLine("###############################");
            Console.WriteLine(result.Url);
            Console.WriteLine(result.Permalink);
            Console.WriteLine(result.ScanId);
            Console.WriteLine(result.Resource);
            Console.WriteLine(result.ResponseCode);
            Console.WriteLine(result.ScanDate);
            Console.WriteLine(result.ScanId);
            Console.WriteLine(result.VerboseMsg);
            Console.WriteLine("###############################");

            return result;
        }

    }
}
