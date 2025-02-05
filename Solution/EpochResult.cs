namespace neuro_app_bep
{
    internal class EpochResult
    {
        public double Loss { get; set; }
        public double Accuracy { get; set; }

        public EpochResult(double loss, double accuracy)
        {
            Loss = loss; Accuracy = accuracy;
        }
    }
}
