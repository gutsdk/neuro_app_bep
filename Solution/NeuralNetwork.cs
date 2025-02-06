using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Linq;

namespace neuro_app_bep
{
    [Serializable]
    public class NeuralNetwork
    {
        [JsonProperty]
        public int InputSize { get; private set; }

        [JsonProperty]
        public int HiddenSize { get; private set; }

        [JsonProperty]
        public int OutputSize { get; private set; }

        [JsonProperty]
        public double LearningRate { get; set; }

        [JsonProperty]
        public double[,] Weights1 { get; private set; }

        [JsonProperty]
        public double[,] Weights2 { get; private set; }

        [JsonProperty]
        public double[] Biases1 { get; private set; }

        [JsonProperty]
        public double[] Biases2 { get; private set; }

        [JsonIgnore]
        private double[] _hiddenInput;

        [JsonIgnore]
        private double[] _hiddenOutput;

        [JsonIgnore]
        private double[] _outputInput;

        [JsonIgnore]
        private double[,] _weights1Grad;

        [JsonIgnore]
        private double[,] _weights2Grad;

        public NeuralNetwork() { }

        public NeuralNetwork(int inputSize, int hiddenSize, int outputSize, double learningRate)
        {
            Initialize(inputSize, hiddenSize, outputSize, learningRate);
        }

        public void Initialize(int inputSize, int hiddenSize, int outputSize, double learningRate)
        {
            InputSize = inputSize;
            HiddenSize = hiddenSize;
            OutputSize = outputSize;
            LearningRate = learningRate;

            var rnd = new Random();
            InitializeWeights(rnd);
            InitializeGradients();
        }

        private void InitializeWeights(Random rnd)
        {
            Weights1 = new double[InputSize, HiddenSize];
            Biases1 = new double[HiddenSize];
            for (int i = 0; i < HiddenSize; i++)
            {
                Biases1[i] = rnd.NextDouble() - 0.5;
                for (int j = 0; j < InputSize; j++)
                {
                    Weights1[j, i] = rnd.NextDouble() - 0.5;
                }
            }

            Weights2 = new double[HiddenSize, OutputSize];
            Biases2 = new double[OutputSize];
            for (int i = 0; i < OutputSize; i++)
            {
                Biases2[i] = rnd.NextDouble() - 0.5;
                for (int j = 0; j < HiddenSize; j++)
                {
                    Weights2[j, i] = rnd.NextDouble() - 0.5;
                }
            }
        }

        private void InitializeGradients()
        {
            _weights1Grad = new double[InputSize, HiddenSize];
            _weights2Grad = new double[HiddenSize, OutputSize];
        }

        public double[] Forward(double[] input)
        {
            _hiddenInput = new double[HiddenSize];
            _hiddenOutput = new double[HiddenSize];

            for (int i = 0; i < HiddenSize; i++)
            {
                _hiddenInput[i] = Biases1[i];
                for (int j = 0; j < InputSize; j++)
                    _hiddenInput[i] += input[j] * Weights1[j, i];

                _hiddenOutput[i] = Math.Max(0, _hiddenInput[i]); // ReLU
            }

            _outputInput = new double[OutputSize];
            for (int i = 0; i < OutputSize; i++)
            {
                _outputInput[i] = Biases2[i];
                for (int j = 0; j < HiddenSize; j++)
                    _outputInput[i] += _hiddenOutput[j] * Weights2[j, i];
            }

            return Softmax(_outputInput);
        }

        public Gradients Backward(double[] input, double[] target, int batchSize)
        {
            var outputGradient = _outputInput.Select((oi, i) => (oi - target[i]) / batchSize).ToArray();

            // Градиенты для weights2 и biases2
            var weights2Grad = new double[HiddenSize, OutputSize];
            var biases2Grad = new double[OutputSize];

            for (int i = 0; i < HiddenSize; i++)
            {
                for (int j = 0; j < OutputSize; j++)
                {
                    weights2Grad[i, j] = outputGradient[j] * _hiddenOutput[i];
                }
            }

            for (int i = 0; i < OutputSize; i++)
            {
                biases2Grad[i] = outputGradient[i];
            }

            // Градиенты для weights1 и biases1
            var hiddenGradient = new double[HiddenSize];
            for (int i = 0; i < HiddenSize; i++)
            {
                var error = 0.0;
                for (int j = 0; j < OutputSize; j++)
                {
                    error += outputGradient[j] * Weights2[i, j];
                }
                hiddenGradient[i] = error * (_hiddenInput[i] > 0 ? 1 : 0);
            }

            var weights1Grad = new double[InputSize, HiddenSize];
            var biases1Grad = new double[HiddenSize];

            for (int i = 0; i < InputSize; i++)
            {
                for (int j = 0; j < HiddenSize; j++)
                {
                    weights1Grad[i, j] = hiddenGradient[j] * input[i];
                }
            }

            for (int i = 0; i < HiddenSize; i++)
            {
                biases1Grad[i] = hiddenGradient[i];
            }

            return new Gradients
            {
                Weights1 = weights1Grad,
                Weights2 = weights2Grad,
                Biases1 = biases1Grad,
                Biases2 = biases2Grad
            };
        }

        public void ApplyGradients(Gradients gradients)
        {
            for (int i = 0; i < InputSize; i++)
            {
                for (int j = 0; j < HiddenSize; j++)
                {
                    Weights1[i, j] -= LearningRate * gradients.Weights1[i, j];
                }
            }

            for (int i = 0; i < HiddenSize; i++)
            {
                for (int j = 0; j < OutputSize; j++)
                {
                    Weights2[i, j] -= LearningRate * gradients.Weights2[i, j];
                }
            }

            for (int i = 0; i < HiddenSize; i++)
            {
                Biases1[i] -= LearningRate * gradients.Biases1[i];
            }

            for (int i = 0; i < OutputSize; i++)
            {
                Biases2[i] -= LearningRate * gradients.Biases2[i];
            }
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
            var target = new double[OutputSize];
            target[label] = 1.0;
            return target;
        }
    }
}