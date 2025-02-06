using Microsoft.Win32;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System.IO;
using System.Windows;
using System.Windows.Media.Imaging;
using Newtonsoft.Json;
using System.Drawing;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace neuro_app_bep
{
    public partial class MainWindow : Window
    {
        private readonly Logger _logger;
        private CancellationTokenSource _trainingCts;
        private readonly object _syncLock = new object();
        private NeuralNetwork _neuralNetwork;
        private TrainingProgress _progress = new TrainingProgress();
        private PlotModel _plotModel;

        public MainWindow()
        {
            InitializeComponent();
            InitializePlot();
            _logger = new Logger("training.log");
            Closing += (s, e) => _trainingCts?.Cancel();
        }

        private void InitializePlot()
        {
            _plotModel = new PlotModel
            {
                Title = "Процесс обучения",
                Background = OxyColors.White
            };

            var lossSeries = new LineSeries
            {
                Title = "Потери",
                Color = OxyColors.Red,
                StrokeThickness = 1
            };

            var accuracySeries = new LineSeries
            {
                Title = "Точность",
                Color= OxyColors.Green,
                StrokeThickness = 1,
                YAxisKey = "Accuracy"
            };

            _plotModel.Axes.Add(new LinearAxis
            {
                Position = AxisPosition.Left,
                Title = "Потери",
                AxislineColor = OxyColors.Black
            });

            _plotModel.Axes.Add(new LinearAxis
            {
                Position = AxisPosition.Right,
                Title = "Точность (%)",
                Key = "Accuracy",
                Minimum = 0, Maximum = 100,
                AxislineColor= OxyColors.Black
            });

            _plotModel.Series.Add(lossSeries);
            _plotModel.Series.Add(accuracySeries);

            Chart.Model = _plotModel;
        }

        private async void Start_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                Start.IsEnabled = false;
                _trainingCts = new CancellationTokenSource();

                await TrainModelAsync(_trainingCts.Token);
            }
            catch (OperationCanceledException)
            {
                _logger.Warning("Обучение прервано пользователем");
                UpdateStatus("Обучение прервано");
            }
            finally
            {
                Start.IsEnabled = true;
            }
        }

        private void PickImage_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "PNG Files (*.png)|*.png|" + "JPG Files (*.jpg)|*.jpg|" +
                    "TIFF Files (*.tiff)|*.tiff|" + "BMP Files (*.bmp)|*.bmp|" + 
                    "All files (*.*)|*.*",
                InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures)
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    var bitmap = new BitmapImage(new Uri(openFileDialog.FileName));
                    PickedImage.Source = bitmap;

                    var processedImage = ProcessImage(bitmap, 28);

                    var output = _neuralNetwork.Forward(processedImage);
                    var softmax = _neuralNetwork.Softmax(output);
                    var predictedDigit = Array.IndexOf(softmax, softmax.Max());

                    MessageBox.Show($"Распознанная цифра: { predictedDigit }\n" +
                        $"Точность: { softmax[predictedDigit] * 100 }%",
                        "Результат распознавания",
                        MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Ошибка распознавания изображения: {ex.Message}",
                        "Ошибка",
                        MessageBoxButton.OK, MessageBoxImage.Error);
                    _logger.Error($"Ошибка: {ex.Message}");
                }
            }
        }

        private void LoadModel_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "JSON files (*.json)|*.json",
                Title = "Выберите файл модели"
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    _logger.Info($"Загрузка модели из {openFileDialog.FileName}");
                    string json = File.ReadAllText(openFileDialog.FileName);
                    _neuralNetwork = JsonConvert.DeserializeObject<NeuralNetwork>(json);

                    ModelStatus.Content = "Модель загружена";
                    MessageBox.Show("Модель успешно загружена!", "Успех",
                                  MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Ошибка загрузки модели: {ex.Message}", "Ошибка",
                                  MessageBoxButton.OK, MessageBoxImage.Error);
                    _logger.Error($"Ошибка: {ex.Message}");
                }
            }
        }

        private void SaveModel_Click(object sender, RoutedEventArgs e)
        {
            if (_neuralNetwork == null)
            {
                MessageBox.Show("Нет обученной модели для сохранения!", "Ошибка",
                              MessageBoxButton.OK, MessageBoxImage.Warning);
                return;
            }

            var saveFileDialog = new SaveFileDialog
            {
                Filter = "JSON files (*.json)|*.json",
                Title = "Сохранить модель"
            };

            if (saveFileDialog.ShowDialog() == true)
            {
                try
                {
                    _logger.Info($"Сохранение модели в {saveFileDialog.FileName}");
                    string json = JsonConvert.SerializeObject(_neuralNetwork);
                    File.WriteAllText(saveFileDialog.FileName, json);

                    MessageBox.Show("Модель успешно сохранена!", "Успех",
                                  MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Ошибка сохранения модели: {ex.Message}", "Ошибка",
                                  MessageBoxButton.OK, MessageBoxImage.Error);
                    _logger.Error($"Ошибка: {ex.Message}");
                }
            }
        }

        private async Task TrainModelAsync(CancellationToken ct)
        {
            try
            {
                var (trainData, testData) = await LoadDataAsync(ct);

                InitializeModel();

                // Конфигурация обучения
                const int epochs = 10;
                const int batchSize = 256;
                var testSample = testData.Take(1000).ToArray();

                // Основной цикл обучения
                for (int epoch = 1; epoch <= epochs; epoch++)
                {
                    ct.ThrowIfCancellationRequested();

                    // Асинхронное обучение с параллельной обработкой
                    var epochResult = await ProcessEpochAsync(
                        trainData,
                        epoch,
                        epochs,
                        batchSize,
                        testSample,
                        ct
                    );

                    // Обновление UI
                    await UpdateTrainingProgressAsync(
                        epochResult.Loss,
                        epochResult.Accuracy,
                        epoch,
                        epochs
                    );
                }


                _logger.Info("Обучение успешно завершено");
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.Error($"Ошибка обучения: {ex}");
                await Dispatcher.InvokeAsync(() =>
                    MessageBox.Show(ex.Message, "Ошибка"));
            }
        }

        private async Task<(List<(double[], int)> train, List<(double[], int)> test)> LoadDataAsync(CancellationToken ct)
        {
            return await Task.Run(() =>
            {
                _logger.Info("Начало загрузки данных");

                var train = MnistLoader.Load("train-images.idx3-ubyte",
                    "train-labels.idx1-ubyte");
                var test = MnistLoader.Load("t10k-images.idx3-ubyte",
                    "t10k-labels.idx1-ubyte");

                ct.ThrowIfCancellationRequested();
                _logger.Info($"Данные загружены: {train.Count} train, {test.Count} test");

                return (train, test);

            }, ct);
        }

        private void InitializeModel()
        {
            Dispatcher.Invoke(() =>
            {
                if (_neuralNetwork == null)
                {
                    _neuralNetwork = new NeuralNetwork();
                    _neuralNetwork.Initialize(784, 256, 10, 0.005);
                    _logger.Info("Модель инициализирована");
                }
            });
        }

        private async Task<EpochResult> ProcessEpochAsync(List<(double[], int)> trainData, int currentEpoch, int totalEpochs, int batchSize,
            (double[], int)[] testSample, CancellationToken ct)
        {
            return await Task.Run(() =>
            {
                var epochWatch = Stopwatch.StartNew();
                double totalLoss = 0;
                var rnd = new Random(Guid.NewGuid().GetHashCode());

                // Параллельное перемешивание данных
                var shuffled = ParallelShuffle(trainData, rnd, ct);

                // Параллельная обработка батчей
                var batches = shuffled
                    .Select((x, i) => new { Index = i, Value = x })
                    .GroupBy(x => x.Index / batchSize)
                    .Select(g => g.Select(x => x.Value).ToList())
                    .ToList();

                Parallel.ForEach(batches, new ParallelOptions
                {
                    CancellationToken = ct,
                    MaxDegreeOfParallelism = Environment.ProcessorCount
                }, batch =>
                {
                    var batchLoss = ProcessBatch(batch, ct);
                    Add(ref totalLoss, batchLoss);
                });

                // Расчет точности
                int correct = CalculateAccuracy(testSample);
                double accuracy = (double)correct / testSample.Length * 100;

                _logger.Info($"Эпоха {currentEpoch}/{totalEpochs} завершена за " +
                           $"{epochWatch.Elapsed.TotalSeconds:F2} сек. " +
                           $"Точность: {accuracy:F2}%");

                return new EpochResult(totalLoss / batches.Count, accuracy);
            }, ct);
        }

        public static double Add(ref double location1, double value)
        {
            double newCurrentValue = location1; // non-volatile read, so may be stale
            while (true)
            {
                double currentValue = newCurrentValue;
                double newValue = currentValue + value;
                newCurrentValue = Interlocked.CompareExchange(ref location1, newValue, currentValue);
                if (newCurrentValue.Equals(currentValue)) 
                    return newValue;
            }
        }

        private List<T> ParallelShuffle<T>(IList<T> list, Random rnd, CancellationToken ct)
        {
            var indices = Enumerable.Range(0, list.Count).ToArray();

            Parallel.For(0, indices.Length, i =>
            {
                int swapIndex = rnd.Next(i, indices.Length);
                (indices[i], indices[swapIndex]) = (indices[swapIndex], indices[i]);
                ct.ThrowIfCancellationRequested();
            });

            return indices.Select(i => list[i]).ToList();
        }

        private int CalculateAccuracy(IEnumerable<(double[] input, int label)> data)
        {
            var correct = new ConcurrentCounter();

            Parallel.ForEach(data, item =>
            {
                var output = _neuralNetwork.Forward(item.input);
                if (Array.IndexOf(output, output.Max()) == item.label)
                    correct.Increment();
            });

            return correct.Value;
        }

        private async Task UpdateTrainingProgressAsync(
            double loss,
            double accuracy,
            int currentEpoch,
            int totalEpochs)
        {
            await Dispatcher.InvokeAsync(() =>
            {
                // Обновление графика
                _progress.Update(loss, accuracy);
                UpdatePlot();

                // Обновление прогресс-бара
                TrainingProgress.Value = (double)currentEpoch / totalEpochs * 100;

                // Обновление текстовых полей
                ModelStatus.Content = $"Эпоха {currentEpoch} из {totalEpochs}";
                AccuracyStatus.Content = $"Точность: {accuracy:F2}%";
            });
        }

        private double[] ProcessImage(BitmapImage image, int resizeValue)
        {
            var greyPixels = new byte[resizeValue * resizeValue];

            using (var memoryStream = new MemoryStream())
            {
                // Универсальная загрузка изображения
                var encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(image));
                encoder.Save(memoryStream);
                memoryStream.Position = 0;

                using (var bmp = new Bitmap(memoryStream))
                using (var resized = new Bitmap(bmp, new System.Drawing.Size(resizeValue, resizeValue)))
                {
                    for (int y = 0; y < resizeValue; y++)
                    {
                        for (int x = 0; x < resizeValue; x++)
                        {
                            var pixel = resized.GetPixel(x, y);
                            byte grayValue = (byte)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                            greyPixels[y * resizeValue + x] = grayValue;
                        }
                    }
                }
            }

            // Добавлена проверка необходимости инверсии
            double average = greyPixels.Average(p => p);
            bool shouldInvert = average > 128; // Если среднее значение яркое, инвертируем
            return greyPixels.Select(p => (shouldInvert ? 255 - p : p) / 255.0).ToArray();
        }

        private void UpdatePlot()
        {
            var lossSeries = (LineSeries)_plotModel.Series[0];
            var accuracySeries = (LineSeries)_plotModel.Series[1];

            lossSeries.Points.Clear();
            accuracySeries.Points.Clear();

            for (int i = 0; i < _progress.LossHistory.Count; i++)
            {
                lossSeries.Points.Add(new DataPoint(i, _progress.LossHistory[i]));
                accuracySeries.Points.Add(new DataPoint(i, _progress.AccuracyHistory[i]));
            }

            _plotModel.InvalidatePlot(true);
        }

        private double ProcessBatch(List<(double[] input, int label)> batch, CancellationToken ct)
        {
            var accumulatedGradients = new Gradients
            {
                Weights1 = new double[_neuralNetwork.InputSize, _neuralNetwork.HiddenSize],
                Weights2 = new double[_neuralNetwork.HiddenSize, _neuralNetwork.OutputSize],
                Biases1 = new double[_neuralNetwork.HiddenSize],
                Biases2 = new double[_neuralNetwork.OutputSize]
            };

            double batchLoss = 0;

            Parallel.ForEach(batch, item =>
            {
                ct.ThrowIfCancellationRequested();

                var (input, label) = item;
                var target = _neuralNetwork.CreateTarget(label);
                var output = _neuralNetwork.Forward(input);
                var gradients = _neuralNetwork.Backward(input, target, batch.Count);

                // Накопление градиентов
                for (int i = 0; i < _neuralNetwork.InputSize; i++)
                {
                    for (int j = 0; j < _neuralNetwork.HiddenSize; j++)
                    {
                        accumulatedGradients.Weights1[i, j] += gradients.Weights1[i, j];
                    }
                }

                for (int i = 0; i < _neuralNetwork.HiddenSize; i++)
                {
                    for (int j = 0; j < _neuralNetwork.OutputSize; j++)
                    {
                        accumulatedGradients.Weights2[i, j] += gradients.Weights2[i, j];
                    }
                }

                for (int i = 0; i < _neuralNetwork.HiddenSize; i++)
                {
                    accumulatedGradients.Biases1[i] += gradients.Biases1[i];
                }

                for (int i = 0; i < _neuralNetwork.OutputSize; i++)
                {
                    accumulatedGradients.Biases2[i] += gradients.Biases2[i];
                }

                batchLoss += _neuralNetwork.CalculateLoss(output, target);
            });

            // Применение градиентов
            _neuralNetwork.ApplyGradients(accumulatedGradients);
            return batchLoss / batch.Count;
        }

        private void UpdateStatus(string message)
        {
            Dispatcher.Invoke(() =>
            {
                ModelStatus.Content = message;
            });
        }
    }
}