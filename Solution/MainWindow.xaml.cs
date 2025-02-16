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
using System.Drawing.Drawing2D;
using MathNet.Numerics.LinearAlgebra;
using System.Globalization;
using System.Collections.Concurrent;
using System.Drawing.Imaging;

namespace neuro_app_bep
{
    public partial class MainWindow : Window
    {
        private readonly Logger _logger;
        private NeuralNetwork _neuralNetwork;
        private PlotModel _plotModel;
        private Stopwatch _trainingTimer = new Stopwatch();
        private CancellationTokenSource _trainingCts;
        private BlockingCollection<TrainingProgressData> _progressQueue = new BlockingCollection<TrainingProgressData>();

        public MainWindow()
        {
            InitializeComponent();
            _logger = new Logger("training.log");
        }

        private void InitializePlot()
        {
            _plotModel = new PlotModel
            {
                Title = "Прогресс обучения",
                Background = OxyColors.White
            };

            _plotModel.Axes.Add(new LinearAxis
            {
                Position = AxisPosition.Left,
                Title = "Потери",
                Key = "LossAxis",
                AxislineColor = OxyColors.Red
            });

            _plotModel.Axes.Add(new LinearAxis
            {
                Position = AxisPosition.Right,
                Title = "Точность (%)",
                Key = "AccuracyAxis",
                Minimum = 0,
                Maximum = 100,
                AxislineColor = OxyColors.Blue
            });

            var lossSeries = new LineSeries
            {
                Title = "Потери",
                Color = OxyColors.Red,
                StrokeThickness = 1,
                YAxisKey = "LossAxis"
            };

            var accuracySeries = new LineSeries
            {
                Title = "Точность",
                Color = OxyColors.Blue,
                StrokeThickness = 1,
                YAxisKey = "AccuracyAxis"
            };

            _plotModel.Series.Add(lossSeries);
            _plotModel.Series.Add(accuracySeries);

            Chart.Model = _plotModel;
        }

        private async void StartButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                _trainingCts = new CancellationTokenSource();
                StartButton.IsEnabled = false;
                CancelButton.IsEnabled = true;

                StatusLabel.Content = "Обучение начато";
                StatsLabel.Content = string.Empty;

                InitializePlot();
                _plotModel.InvalidatePlot(true);

                // Запуск обработки данных в фоновом потоке
                var trainingTask = Task.Run(() => TrainModelAsync(_trainingCts.Token), _trainingCts.Token);

                // Запуск обновления UI
                var uiUpdateTask = UpdatePlotAsync(_trainingCts.Token);

                await Task.WhenAll(trainingTask, uiUpdateTask);
            }
            catch (OperationCanceledException)
            {
                _logger.Warning("Обучение прервано пользователем");
            }
            finally
            {
                StatusLabel.Content = "Обучение завершено";
                StartButton.IsEnabled = true;
                CancelButton.IsEnabled = false;
            }
        }

        private void CancelButton_Click(object sender, RoutedEventArgs e)
        {
            _trainingCts?.Cancel();
            StatusLabel.Content = "Обучение отменено";
        }

        private void PickImageButton_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "Изображения|*.png;*.jpg;*.jpeg;*.tiff;*.bmp|Все файлы|*.*",
                InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures)
            };

            if (openFileDialog.ShowDialog() == true)
            {
                try
                {
                    var bitmap = new BitmapImage(new Uri(openFileDialog.FileName));
                    PickedImage.Source = bitmap;

                    var processedImage = ProcessImage(bitmap, 28);
                    var (recognizedDigit, confidence) = _neuralNetwork.Predict(processedImage);

                    MessageBox.Show($"Распознанная цифра: {recognizedDigit}\n" +
                        $"Точность: {confidence:P2}",
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

        private void LoadModelButton_Click(object sender, RoutedEventArgs e)
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

                    var settings = new JsonSerializerSettings();
                    settings.Converters.Add(new MatrixConverter());
                    settings.Converters.Add(new VectorConverter());

                    _neuralNetwork = JsonConvert.DeserializeObject<NeuralNetwork>(json, settings);

                    StatusLabel.Content = "Модель загружена";
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

        private void SaveModelButton_Click(object sender, RoutedEventArgs e)
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

                    var settings = new JsonSerializerSettings();
                    settings.Converters.Add(new MatrixConverter());
                    settings.Converters.Add(new VectorConverter());

                    string json = JsonConvert.SerializeObject(_neuralNetwork, Formatting.Indented, settings);
                    File.WriteAllText(saveFileDialog.FileName, json);

                    StatusLabel.Content = "Модель сохранена";
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
                const int batchSize = 128;
                const double learningRate = 0.001;
                const int epochs = 50;

                // Загрузка данных в матричном формате
                var (trainX, trainY) = MnistLoader.LoadMatrix("train");
                var (testX, testY) = MnistLoader.LoadMatrix("t10k");

                InitializeModel(learningRate);
                _logger.Info("Модель инициализирована");

                for (int epoch = 1; epoch <= epochs; epoch++)
                {
                    _trainingTimer.Restart();
                    ct.ThrowIfCancellationRequested();

                    // Перемешивание данных
                    var (shuffledX, shuffledY) = Shuffle(trainX, trainY);

                    // Обучение по батчам
                    for (int i = 0; i < shuffledX.RowCount; i += batchSize)
                    {
                        int currentBatchSize = Math.Min(batchSize, shuffledX.RowCount - i);

                        var batchX = shuffledX.SubMatrix(i, currentBatchSize, 0, _neuralNetwork.InputSize);
                        var batchY = shuffledY.SubMatrix(i, currentBatchSize, 0, _neuralNetwork.OutputSize);

                        var output = _neuralNetwork.Forward(batchX);
                        _neuralNetwork.Backward(batchX, batchY, output);
                    }

                    // Валидация
                    var testOutput = _neuralNetwork.Forward(testX);
                    var accuracy = CalculateAccuracy(testOutput, testY);
                    var loss = CalculateLoss(testOutput, testY);

                    _progressQueue.Add(new TrainingProgressData
                    {
                        Epoch = epoch,
                        Loss = loss,
                        Accuracy = accuracy
                    });

                    await Task.Delay(100, ct); // Имитация задержки
                }
                _progressQueue.CompleteAdding();
                _logger.Info("Обучение успешно завершено");
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                _logger.Error($"Ошибка обучения: {ex}");
                await Dispatcher.InvokeAsync(() =>
                    MessageBox.Show(ex.Message, "Ошибка"));
            }
        }

        private void InitializeModel(double learningRate)
        {
            Dispatcher.Invoke(() =>
            {
                if (_neuralNetwork == null)
                {
                    _neuralNetwork = new NeuralNetwork(784, 256, 10, learningRate);
                }
            });
        }

        private (Matrix<double>, Matrix<double>) Shuffle(Matrix<double> x, Matrix<double> y)
        {
            // Проверка согласованности данных
            if (x.RowCount != y.RowCount)
                throw new ArgumentException("Количество примеров в X и Y не совпадает");

            // Оптимизированный алгоритм перемешивания Фишера-Йетса
            var indices = Enumerable.Range(0, x.RowCount).ToArray();
            var rng = new Random();

            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            // Создание матриц без материализации всех строк
            return (
                Matrix<double>.Build.DenseOfRows(indices.Select(x.Row)),
                Matrix<double>.Build.DenseOfRows(indices.Select(y.Row))
            );
        }

        private double CalculateAccuracy(Matrix<double> output, Matrix<double> y)
        {
            return output.EnumerateRows()
                .Zip(y.EnumerateRows(), (p, t) =>
                    p.MaximumIndex() == t.MaximumIndex() ? 1 : 0)
                .Average() * 100;
        }

        private double CalculateLoss(Matrix<double> output, Matrix<double> y)
        {
            return output.EnumerateRows()
                .Zip(y.EnumerateRows(), (p, t) =>
                    -p.Enumerate().Zip(t.Enumerate(), (pi, ti) => ti * Math.Log(pi + 1e-10)).Sum())
                .Average();
        }

        private async Task UpdatePlotAsync(CancellationToken ct)
        {
            await Task.Run(() =>
            {
                foreach (var data in _progressQueue.GetConsumingEnumerable(ct))
                {
                    Dispatcher.Invoke(() =>
                    {
                        var lossSeries = (LineSeries)_plotModel.Series[0];
                        var accuracySeries = (LineSeries)_plotModel.Series[1];

                        lossSeries.Points.Add(new DataPoint(data.Epoch, data.Loss));
                        accuracySeries.Points.Add(new DataPoint(data.Epoch, data.Accuracy));
                        
                        _plotModel.InvalidatePlot(true);

                        // Обновление статуса
                        StatsLabel.Content = $"Эпоха: {data.Epoch} | Потери: {data.Loss:F4} | Точность: {data.Accuracy:F2}% | Время: {_trainingTimer.Elapsed.TotalSeconds:F2}с";
                    });
                }
            }, ct);
        }

        private double[] ProcessImage(BitmapImage image, int resizeValue)
        {
            var pixels = new double[resizeValue * resizeValue];

            using (var memoryStream = new MemoryStream())
            {
                // Сохраняем BitmapImage в поток
                var encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(image));
                encoder.Save(memoryStream);
                memoryStream.Position = 0;

                using (var bmp = new Bitmap(memoryStream))
                using (var resized = new Bitmap(bmp, new System.Drawing.Size(resizeValue, resizeValue)))
                {
                    var rect = new Rectangle(0, 0, resizeValue, resizeValue);
                    // Захватываем данные изображения в формате 24bppRgb
                    var bmpData = resized.LockBits(rect, ImageLockMode.ReadOnly, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

                    int stride = bmpData.Stride;
                    int bytesCount = stride * resizeValue;
                    byte[] rgbValues = new byte[bytesCount];

                    System.Runtime.InteropServices.Marshal.Copy(bmpData.Scan0, rgbValues, 0, bytesCount);

                    resized.UnlockBits(bmpData);

                    double sum = 0;

                    for (int y = 0; y < resizeValue; y++)
                    {
                        for (int x = 0; x < resizeValue; x++)
                        {
                            int index = y * stride + x * 3;
                            byte blue = rgbValues[index];
                            byte green = rgbValues[index + 1];
                            byte red = rgbValues[index + 2];

                            byte gray = (byte)(0.299 * red + 0.587 * green + 0.114 * blue);
                            pixels[y * resizeValue + x] = gray;
                            sum += gray;
                        }
                    }

                    double avg = sum / (resizeValue * resizeValue);

                    // Нормализация
                    if (avg > 128)
                    {
                        for (int i = 0; i < pixels.Length; i++)
                            pixels[i] = (255 - pixels[i]) / 255.0;
                    }
                    else
                    {
                        for (int i = 0; i < pixels.Length; i++)
                            pixels[i] = pixels[i] / 255.0;
                    }
                }
            }
            return pixels;
        }
    }
}