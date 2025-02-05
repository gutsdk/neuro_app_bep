namespace neuro_app_bep
{
    internal class ConcurrentCounter
    {
        private int _value;
        public int Value => _value;

        public void Increment() => Interlocked.Increment(ref _value);
    }
}
