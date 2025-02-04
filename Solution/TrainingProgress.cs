namespace neuro_app_bep
{
    internal class TrainingProgress
    {
        public List<double> LossHistory { get; } = new List<double>();
        public List<double> AccuracyHistory { get; } = new List<double>();

        public void Update(double loss, double accuracy)
        {
            LossHistory.Add(loss);
            AccuracyHistory.Add(accuracy);
        }
    }
}
