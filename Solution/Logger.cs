using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;

namespace neuro_app_bep
{
    internal class Logger : IDisposable
    {
        private readonly string _logFilePath;
        private readonly BlockingCollection<string> _logQueue = new BlockingCollection<string>();
        private readonly Task _loggingTask;

        public Logger(string filePath)
        {
            _logFilePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, filePath);
            _loggingTask = Task.Run(ProcessLogQueue);
        }

        public void Info(string message) => Log("INFO", message);
        public void Warning(string message) => Log("WARNING", message);
        public void Error(string message) => Log("ERROR", message);

        private void Log(string level, string message)
        {
            var logEntry = $"{DateTime.Now:dd-MM-yyyy HH:mm:ss} [{ level }] { message }";
            _logQueue.Add(logEntry);
        }

        private void ProcessLogQueue()
        {
            try
            {
                using (var writer = new StreamWriter(_logFilePath, true))
                {
                    foreach (var logEntry in _logQueue.GetConsumingEnumerable())
                    {
                        writer.WriteLine(logEntry);
                        Debug.WriteLine(logEntry);
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Ошибка логирования: { ex }");
            }
        }

        public void Dispose()
        {
            _logQueue.CompleteAdding();
            _loggingTask.Wait();
        }
    }
}
