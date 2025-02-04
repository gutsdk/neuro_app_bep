using System.IO;
using System.Net;
using System.Windows;

namespace neuro_app_bep
{
    public static class MnistLoader
    {
        public static List<(double[], int)> Load(string imagesPath, string labelsPath)
        {
            var images = new List<double[]>();
            using (var stream = File.Open(imagesPath, FileMode.Open))
            using (var reader = new BinaryReader(stream))
            {
                reader.ReadInt32();                                             // Магическое число
                int count = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                int rows = IPAddress.NetworkToHostOrder(reader.ReadInt32());
                int cols = IPAddress.NetworkToHostOrder(reader.ReadInt32());

                for (int i = 0; i < count; i++)
                {
                    var bytes = reader.ReadBytes(rows * cols);
                    var image = bytes.Select(b => (double)b / 255).ToArray();
                    images.Add(image);
                }
            }

            var labels = new List<int>();
            using (var stream = File.Open(labelsPath, FileMode.Open))
            using (var reader = new BinaryReader(stream))
            {
                reader.ReadInt32();                                             // Магическое число
                int count = IPAddress.NetworkToHostOrder(reader.ReadInt32());

                for (int i = 0; i < count; i++)
                    labels.Add(reader.ReadByte());
            }

            return images.Zip(labels, (i, l) => (i, l)).ToList();
        }
    }
}
