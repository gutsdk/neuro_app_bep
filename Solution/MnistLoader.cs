using MathNet.Numerics.LinearAlgebra;
using System.Drawing.Drawing2D;
using System.IO;
using System.Net;
using System.Windows;

namespace neuro_app_bep
{
    public static class MnistLoader
    {
        public static (Matrix<double>, Matrix<double>) LoadMatrix(string dataset)
        {
            var images = LoadImages($"{dataset}-images.idx3-ubyte"); // Должен вернуть List<byte[]>
            var labels = LoadLabels($"{dataset}-labels.idx1-ubyte"); // Должен вернуть List<int>

            int sampleCount = images.Count;
            int featureSize = 784; // 28x28 изображения
            int numClasses = 10;   // Классы от 0 до 9

            var X = Matrix<double>.Build.Dense(sampleCount, featureSize);
            var Y = Matrix<double>.Build.Dense(sampleCount, numClasses);

            Parallel.For(0, sampleCount, i =>
            {
                X.SetRow(i, images[i].Select(v => (double)v / 255.0).ToArray()); // Нормализация
                Y[i, labels[i]] = 1.0; // One-hot encoding
            });

            return (X, Y);
        }

        public static List<byte[]> LoadImages(string path)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs);

            int magicNumber = br.ReadInt32BigEndian();
            if (magicNumber != 0x00000803)
                throw new Exception("Неверный формат файла изображений.");

            int numImages = br.ReadInt32BigEndian();
            int height = br.ReadInt32BigEndian();
            int width = br.ReadInt32BigEndian();
            int imageSize = height * width;

            var images = new List<byte[]>(numImages);
            for (int i = 0; i < numImages; i++)
                images.Add(br.ReadBytes(imageSize));

            return images;
        }

        public static List<int> LoadLabels(string path)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
            using var br = new BinaryReader(fs);

            int magicNumber = br.ReadInt32BigEndian();
            if (magicNumber != 0x00000801)
                throw new Exception("Неверный формат файла меток.");

            int numLabels = br.ReadInt32BigEndian();
            var labels = new List<int>(numLabels);

            for (int i = 0; i < numLabels; i++)
                labels.Add(br.ReadByte());

            return labels;
        }

        private static int ReadInt32BigEndian(this BinaryReader br)
        {
            var bytes = br.ReadBytes(4);
            if (BitConverter.IsLittleEndian)
                Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
