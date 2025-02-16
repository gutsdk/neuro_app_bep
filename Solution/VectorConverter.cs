using Newtonsoft.Json;
using MathNet.Numerics.LinearAlgebra;

namespace neuro_app_bep
{
    public class VectorConverter : JsonConverter<Vector<double>>
    {
        public override Vector<double>? ReadJson(JsonReader reader, Type objectType, Vector<double>? existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            var data = serializer.Deserialize<double[]>(reader);
            return Vector<double>.Build.Dense(data);
        }

        public override void WriteJson(JsonWriter writer, Vector<double>? value, JsonSerializer serializer)
        {
            serializer.Serialize(writer, value.ToArray());
        }
    }
}
