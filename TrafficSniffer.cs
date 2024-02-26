using System.Net.Sockets;
using System.Net;
using System.Diagnostics;
using System.Xml.XPath;

using VirusTotalNet;
using VirusTotalNet.Results;

namespace WebSniffer
{
    public class TrafficSniffer
    {
        public static List<PacketJson> packetCollect { get; set; } = new List<PacketJson>();
        public class PacketJson
        {
            public DateTime RecordTime { get; set; }
            public string OrgData { get; set; }
            public string SrcAddr { get; set; }
            public string DstAddr { get; set; }
            public string Protocol { get; set; }
            public string Payload { get; set; }
        }

        public static async Task MainAsync(int milliseconds = 10000)
        {
            Socket socket = new Socket(AddressFamily.InterNetwork, SocketType.Raw, ProtocolType.IP);
            socket.Bind(new IPEndPoint(IPAddress.Parse("0.0.0.0"), 0));
            socket.IOControl(IOControlCode.ReceiveAll, BitConverter.GetBytes(1), null);

            byte[] buffer = new byte[65535];
            int bytesRead;

            Stopwatch st = new Stopwatch();
            st.Start();
            while (st.ElapsedMilliseconds < milliseconds)
            {
                bytesRead = socket.Receive(buffer, 0, buffer.Length, SocketFlags.None);
                // Process the packet data here
                ProcessPacketBdy(buffer, bytesRead);
            }
            st.Stop();
        }

        static void ProcessPacketBdy(byte[] buffer, int length)
        {
            // Assuming the packet is an IP packet
            if (length >= 20) // Minimum length for an IPv4 header
            {
                byte versionAndHeaderLength = buffer[0];
                int version = versionAndHeaderLength >> 4;
                int headerLength = (versionAndHeaderLength & 0xF) * 4;

                if (version == 4) // IPv4
                {
                    byte protocol = buffer[9]; // Protocol field in the IPv4 header

                    if (protocol == 6 || protocol == 17) // TCP
                    {
                        string protocolStr = protocol == 6 ? "TCP" : "UDP";
                        // Extract TCP header information
                        ushort sourcePort = BitConverter.ToUInt16(buffer, headerLength);
                        ushort destinationPort = BitConverter.ToUInt16(buffer, headerLength + 2);

                        int dataOffset = ((buffer[headerLength + 12] & 0xF0) >> 4) * 4; // TCP header length
                        int dataLength = length - headerLength - dataOffset;

                        if (dataLength <= 0)
                            return;

                        byte[] payload = new byte[dataLength];

                        if (payload.Length == 0)
                            return;
                        Buffer.BlockCopy(buffer, headerLength + dataOffset, payload, 0, dataLength);
                        IPAddress sourceAddress = new IPAddress(BitConverter.ToUInt32(buffer, 12));
                        IPAddress destinationAddress = new IPAddress(BitConverter.ToUInt32(buffer, 16));
                        // Now you can analyze/process the payload (application layer data)
                        Console.WriteLine("###############################");
                        Console.WriteLine($"IPv4 Packet - Source: {sourceAddress}, Destination: {destinationAddress}, Protocol: {protocol}");
                        Console.WriteLine($"{protocolStr} Packet - Source Port: {sourcePort}, Destination Port: {destinationPort}, Length : {dataLength}");
                        //Console.WriteLine($"All Data: {BitConverter.ToString(buffer.ToList().Where(x=>x!=0).ToArray())}");
                        Console.WriteLine($"Payload Data: {BitConverter.ToString(payload)}");
                        Console.WriteLine("###############################");
                        Console.WriteLine();

                        packetCollect.Add(new PacketJson()
                        {
                            RecordTime = DateTime.Now,
                            OrgData = BitConverter.ToString(buffer.ToList().Where(x => x != 0).ToArray()),
                            SrcAddr = sourceAddress.ToString(),
                            DstAddr = destinationAddress.ToString(),
                            Payload = BitConverter.ToString(payload),
                            Protocol = protocol.ToString(),
                        });
                    }
                }
            }
        }

    }
}
