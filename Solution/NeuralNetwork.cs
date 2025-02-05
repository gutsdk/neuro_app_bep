using Newtonsoft.Json;

namespace neuro_app_bep
{
    public class NeuralNetwork
    {
        public int _inputSize;
        public int _hiddenSize;
        public int _outputSize;
        public double _learningRate;

        public double[,] _weights1;
        public double[,] _weights2;
        public double[] _biases1;
        public double[] _biases2;

        [JsonIgnore]
        public double[] _hiddenInput;
        [JsonIgnore]
        public double[] _hiddenOutput;
        [JsonIgnore]
        public double[] _outputInput;
        [JsonIgnore]
        public double[,] weights1Grad;
        [JsonIgnore]
        public double[,] weights2Grad;

        public void Initialize(int inputSize, int hiddenSize, int outputSize, double learningRate)
        {
            _inputSize = inputSize;
            _hiddenSize = hiddenSize;
            _outputSize = outputSize;
            _learningRate = learningRate;

            var rnd = new Random();

            // весовые коэффициенты
            _weights1 = new double[inputSize, hiddenSize];
            _biases1 = new double[hiddenSize];
            for (int i = 0; i < hiddenSize; i++)
            {
                _biases1[i] = rnd.NextDouble() - 0.5;
                for (int j = 0; j < inputSize; j++)
                {
                    _weights1[j, i] = rnd.NextDouble() - 0.5;
                }
            }

            _weights2 = new double[hiddenSize, outputSize];
            _biases2 = new double[outputSize];
            for (int i = 0; i < outputSize; i++)
            {
                _biases2[i] = rnd.NextDouble() - 0.5;
                for (int j = 0; j < hiddenSize; j++)
                {
                    _weights2[j, i] = rnd.NextDouble() - 0.5;
                }
            }
        }

        public double[] Forward(double[] input)
        {
            _hiddenInput = new double[_hiddenSize];
            _hiddenOutput = new double[_hiddenSize];

            for (int i = 0; i <  _hiddenSize; i++)
            {
                _hiddenInput[i] = _biases1[i];
                for (int j = 0; j < _inputSize; j++)
                    _hiddenInput[i] += input[j] * _weights1[j, i];

                _hiddenOutput[i] = Math.Max(0, _hiddenInput[i]); // ReLU
            }

            _outputInput = new double[_outputSize];
            for (int i = 0; i < _outputSize; i++)
            {
                _outputInput[i] = _biases2[i];
                for (int j = 0; j < _hiddenSize; j++)
                    _outputInput[i] += _hiddenOutput[j] * _weights2[j, i];
            }

            var output = Softmax(_outputInput);
            return output;
        }

        public void Backward(double[] input, double[] target, int batchSize)
        {
            var outputGradient = _outputInput.Select((oi, i) => (oi - target[i]) / batchSize).ToArray();

            for (int i = 0; i < _hiddenSize; i++)
                for (int j =  i; j < _outputSize; j++)
                    _weights2[i, j] -= _learningRate * outputGradient[j] * _hiddenOutput[i];

            for (int i = 0; i < _outputSize; i++)
                _biases2[i] -= _learningRate * outputGradient[i];


            var hiddenGradient = new double[_hiddenSize];
            for (int i = 0; i < _hiddenSize; i++)
            {
                var error = 0.0;
                for (int j = 0; j < _outputSize; j++)
                    error += outputGradient[j] * _weights2[i, j];

                hiddenGradient[i] = error * (_hiddenInput[i] > 0 ? 1 : 0);
            }

            for (int i = 0; i < _inputSize; i++)
                for (int j = 0; j < _hiddenSize; j++)
                    _weights1[i, j] -= _learningRate * hiddenGradient[j] * input[i];

            for (int i = 0; i < _hiddenSize; i++)
                _biases1[i] -= _learningRate * hiddenGradient[i];
        }

        public double[] Softmax(double[] input)
        {
            var exp = input.Select(x => Math.Exp(x - input.Max())).ToArray();
            var sum = exp.Sum();
            return exp.Select(x => x / sum).ToArray();
        }

        public double CalculateLoss(double[] output, double[] target)
        {
            double loss = 0;
            for (int i = 0; i < output.Length; i++)
                loss += -target[i] * Math.Log(output[i] + 1e-10);

            return loss;
        }

        public double[] CreateTarget(int label)
        {
            var target = new double[10]; // выходные данные 0-9 в сумме 10
            target[label] = 1.0;
            return target;
        }

        public void MergeGradients(Dictionary<int, double[]> localGradients)
        {
            // Реализация объединения градиентов
            foreach (var key in localGradients.Keys)
            {
                for (int i = 0; i < localGradients[key].Length; i++)
                {
                    weights1Grad[key / _hiddenSize, key % _hiddenSize] += localGradients[key][i];
                }
            }
        }
    }
}
