using Newtonsoft.Json;
using MathNet.Numerics.LinearAlgebra;

namespace neuro_app_bep
{
    public class MatrixConverter : JsonConverter<Matrix<double>>
    {
        public override Matrix<double>? ReadJson(JsonReader reader, Type objectType, Matrix<double>? existingValue, bool hasExistingValue, JsonSerializer serializer)
        {
            var data = serializer.Deserialize<double[][]>(reader);
            return Matrix<double>.Build.DenseOfRows(data);
        }

        public override void WriteJson(JsonWriter writer, Matrix<double>? value, JsonSerializer serializer)
        {
            serializer.Serialize(writer, value.ToRowArrays());
        }
    }
}
