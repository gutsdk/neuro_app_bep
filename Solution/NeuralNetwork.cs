using Newtonsoft.Json;
using MathNet.Numerics.LinearAlgebra;

namespace neuro_app_bep
{
    [Serializable]
    public class NeuralNetwork
    {
        [JsonProperty]
        public int InputSize { get; set; }

        [JsonProperty]
        public int HiddenSize { get; set; }

        [JsonProperty]
        public int OutputSize { get; set; }

        [JsonProperty]
        public double LearningRate { get; set; }

        [JsonProperty]
        public Matrix<double> Weights1 { get; set; }

        [JsonProperty]
        public Matrix<double> Weights2 { get; set; }

        [JsonProperty]
        public Vector<double> Biases1 { get; set; }

        [JsonProperty]
        public Vector<double> Biases2 { get; set; }

        [JsonIgnore]
        private Matrix<double> _A1;

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate)
        {
            InputSize = inputSize;
            HiddenSize = hiddenSize;
            OutputSize = outputSize;
            LearningRate = learningRate;


            InitializeWeights();
        }

        private void InitializeWeights()
        {
            var rnd = new Random();

            // Инициализация для ReLU
            Weights1 = Matrix<double>.Build.Random(InputSize, HiddenSize,
                new MathNet.Numerics.Distributions.Normal(0, Math.Sqrt(2.0 / InputSize)));

            Biases1 = Vector<double>.Build.Dense(HiddenSize);

            // Инициализация для выходного слоя
            Weights2 = Matrix<double>.Build.Random(HiddenSize, OutputSize,
                new MathNet.Numerics.Distributions.Normal(0, Math.Sqrt(1.0 / HiddenSize)));

            Biases2 = Vector<double>.Build.Dense(OutputSize);
        }

        public Matrix<double> Forward(Matrix<double> X)
        {
            var Z1 = X * Weights1;
            for (int i = 0; i < Z1.RowCount; i++)
                Z1.SetRow(i, Z1.Row(i) + Biases1);

            _A1 = Z1.Map(ReLU);
            var Z2 = _A1 * Weights2;

            for (int i = 0; i < Z2.RowCount; i++)
                Z2.SetRow(i, Z2.Row(i) + Biases2);

            return Softmax(Z2);
        }

        public void Backward(Matrix<double> X, Matrix<double> Y, Matrix<double> output)
        {
            var batchSize = X.RowCount;

            var dZ2 = output - Y;
            var dW2 = (_A1.Transpose() * dZ2) / batchSize;
            var dB2 = dZ2.ColumnSums() / batchSize;

            var dZ1 = (dZ2 * Weights2.Transpose()).PointwiseMultiply(_A1.Map(ReLUDerivative));
            var dW1 = (X.Transpose() * dZ1) / batchSize;
            var dB1 = dZ1.ColumnSums() / batchSize;

            Weights2 -= LearningRate * dW2;
            Weights1 -= LearningRate * dW1;
            Biases2 -= LearningRate * dB2;
            Biases1 -= LearningRate * dB1;
        }

        public Matrix<double> Softmax(Matrix<double> input)
        {
            var maxPerRow = Vector<double>.Build.Dense(input.RowCount, i => input.Row(i).AbsoluteMaximum());
            var stabilized = input - Matrix<double>.Build.Dense(input.RowCount, input.ColumnCount, (i, j) => maxPerRow[i]);

            var exp = stabilized.PointwiseExp();
            var sum = exp.RowSums();
            return exp.MapIndexed((i, j, v) => v / sum[i]);
        }

        private double ReLU(double x) => Math.Max(0, x);
        private double ReLUDerivative(double x) => x > 0 ? 1 : 0;

        public (int digit, double confidence) Predict(double[] input)
        {
            var inputMatrix = Matrix<double>.Build.DenseOfRowArrays(input);
            var output = Forward(inputMatrix); // Прогоняем через сеть

            // Получаем индекс максимального значения (это предсказанная цифра)
            int predictedDigit = output.Row(0).MaximumIndex();
            double confidence = output[0, predictedDigit]; // Вероятность

            return (predictedDigit, confidence);
        }
    }
}