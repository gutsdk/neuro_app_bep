using Microsoft.Win32;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System.IO;
using System.Text.Json;
using System.Windows;
using System.Windows.Media.Imaging;

namespace neuro_app_bep
{
    public partial class MainWindow : Window
    {
        private PlotModel _plotModel;
        private NeuralNetwork _neuralNetwork;
        private TrainingProgress _progress = new TrainingProgress();
        public MainWindow()
        {
            InitializeComponent();
            InitializePlot();
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
            Start.IsEnabled = false;
            ModelStatus.Content = "Обучение...";

            await Task.Run(() => TrainModel());

            ModelStatus.Content = "Обучение завершено";
            Start.IsEnabled = true;
        }

        private void PickImage_Click(object sender, RoutedEventArgs e)
        {
            var openFileDialog = new OpenFileDialog
            {
                Filter = "PNG Files (*.png)|*.png|All files (*.*)|*.*",
                InitialDirectory = Environment.GetFolderPath(Environment.SpecialFolder.MyPictures)
            };

            if (openFileDialog.ShowDialog() == true)
            {
                var bitmap = new BitmapImage(new Uri(openFileDialog.FileName));
                PickedImage.Source = bitmap;
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
                    string json = File.ReadAllText(openFileDialog.FileName);
                    _neuralNetwork = JsonSerializer.Deserialize<NeuralNetwork>(json);

                    ModelStatus.Content = "Модель загружена";
                    MessageBox.Show("Модель успешно загружена!", "Успех",
                                  MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Ошибка загрузки модели: {ex.Message}", "Ошибка",
                                  MessageBoxButton.OK, MessageBoxImage.Error);
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
                    var options = new JsonSerializerOptions { WriteIndented = true }; // Красивый JSON
                    string json = JsonSerializer.Serialize(_neuralNetwork, options);
                    File.WriteAllText(saveFileDialog.FileName, json);

                    MessageBox.Show("Модель успешно сохранена!", "Успех",
                                  MessageBoxButton.OK, MessageBoxImage.Information);
                }
                catch (Exception ex)
                {
                    MessageBox.Show($"Ошибка сохранения модели: {ex.Message}", "Ошибка",
                                  MessageBoxButton.OK, MessageBoxImage.Error);
                }
            }
        }

        private void TrainModel()
        {
            try
            {
                var trainData = MnistLoader.Load("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
                var testData = MnistLoader.Load("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");

                if (_neuralNetwork == null)
                {
                    _neuralNetwork = new NeuralNetwork();
                    _neuralNetwork.Initialize(784, 128, 10, 0.01);
                }

                int epochs = 10;
                int batchSize = 32;

                for (int epoch = 0; epoch < epochs; epoch++)
                {
                    double epochLoss = 0;
                    var rnd = new Random();
                    var shuffled = trainData.OrderBy(x => rnd.Next()).ToList();

                    for (int i = 0; i < shuffled.Count; i += batchSize)
                    {
                        var batch = shuffled.Skip(i).Take(batchSize).ToList();
                        double batchLoss = 0;

                        foreach (var (input, label) in batch)
                        {
                            var output = _neuralNetwork.Forward(input);
                            var target = new double[10];
                            target[label] = 1.0;
                            _neuralNetwork.Backward(input, target, batch.Count);
                            batchLoss += _neuralNetwork.CalculateLoss(output, target);
                        }

                        epochLoss += batchLoss / batch.Count;
                    }

                    // Тестирование
                    int correct = 0;
                    foreach (var (input, label) in testData.Take(1000))
                    {
                        var output = _neuralNetwork.Forward(input);
                        if (output.ToList().IndexOf(output.Max()) == label)
                            correct++;
                    }

                    double accuracy = correct / 10.0;
                    _progress.Update(epochLoss / (trainData.Count / batchSize), accuracy);

                    // Обновление UI
                    Dispatcher.Invoke(() =>
                    {
                        AccuracyStatus.Content = $"Точность: {accuracy:F1}%";
                        UpdatePlot();
                    });
                }
            }
            catch (FileNotFoundException ex)
            {
                MessageBox.Show("Скачайте датасет mnist и расположите вместе с исполняющим файлом");
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString());
            }
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
    }
}