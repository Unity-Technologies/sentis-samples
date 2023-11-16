using System;
using Unity.Sentis;

namespace Unity.Sentis
{
    public partial class CPUBackend
    {
        internal static void EinsumND(TensorFloat[] inputTensors, TensorFloat O, TensorShape[] operandShapes, TensorIndex[] operandIndices, TensorIndex outputIndices, TensorShape outputShape, TensorIndex sumIndices, TensorShape sumShape, int numIndices)
        {
            for (var i = 0; i < inputTensors.Length; i++)
            {
                ArrayTensorData.Pin(inputTensors[i]);
            }

            ArrayTensorData.Pin(O, clearOnInit: false);

            var outSize = O.shape.length;
            var sumSize = sumShape.length;

            Span<int> position = stackalloc int[outputIndices.rank + sumIndices.rank];

            for (var outIndex = 0; outIndex < outSize; outIndex++)
            {
                SetPositionFromIndex(position, outputIndices, outputShape, outIndex);
                float sum = 0;
                for (var sumIndex = 0; sumIndex < sumSize; sumIndex++)
                {
                    SetPositionFromIndex(position, sumIndices, sumShape, sumIndex);
                    float product = 1f;
                    for (var i = 0; i < inputTensors.Length; i++)
                    {
                        var operandIndex = GetIndexFromPosition(position, operandIndices[i], inputTensors[i].shape);
                        product *= inputTensors[i][operandIndex];
                    }

                    sum += product;
                }

                O[outIndex] = sum;
            }
        }

        static void SetPositionFromIndex(Span<int> position, TensorIndex indices, TensorShape shape, int index)
        {
            for (var i = shape.rank - 1; i >= 0; i--)
            {
                position[indices[i]] = index % shape[i];
                index /= shape[i];
            }
        }

        static int GetIndexFromPosition(Span<int> position, TensorIndex indices, TensorShape shape)
        {
            var index = 0;
            var stride = 1;
            for (var i = shape.rank - 1; i >= 0; i--)
            {
                index += stride * position[indices[i]];
                stride *= shape[i];
            }
            return index;
        }
    }
}
