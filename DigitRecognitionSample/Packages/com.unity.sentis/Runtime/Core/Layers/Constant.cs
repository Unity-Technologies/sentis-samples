using System;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents a constant in a model.
    /// </summary>
    public class Constant
    {
        /// <summary>
        /// The name of the constant.
        /// </summary>
        public string name;
        /// <summary>
        /// The shape of the constant as a `TensorShape`.
        /// </summary>
        public TensorShape shape;
        /// <summary>
        /// The number of elements in the constant.
        /// </summary>
        public Int32 length;
        /// <summary>
        /// The offset of the first element of the constant in the `weights` array.
        /// </summary>
        public Int64 offset;
        /// <summary>
        /// The data type of the constant as a `DataType`.
        /// </summary>
        public DataType dataType;
        /// <summary>
        /// The elements of the constant as a `NativeTensorArray`.
        /// </summary>
        public NativeTensorArray weights;

        /// <summary>
        /// Initializes and returns an empty `Constant`.
        /// </summary>
        public Constant() { }

        /// <summary>
        /// Initializes and returns a `Constant` from a given name and tensor.
        /// </summary>
        /// <param name="name">The name to use for the constant.</param>
        /// <param name="tensor">The tensor to take the shape, dataType and weights of the constant from.</param>
        public Constant(string name, Tensor tensor)
        {
            this.name = name;
            this.offset = 0;
            this.shape = tensor.shape;
            this.length = tensor.shape.length;
            this.dataType = tensor.dataType;
            if (tensor.shape.length == 0)
                return;
            weights = new NativeTensorArray(tensor.shape.length);

            switch (dataType)
            {
                case DataType.Float:
                {
                    NativeTensorArray.Copy(tensor.ToReadOnlyNativeArray<float>(), 0, weights, 0, weights.Length);
                    break;
                }
                case DataType.Int:
                {
                    NativeTensorArray.Copy(tensor.ToReadOnlyNativeArray<int>(), 0, weights, 0, weights.Length);
                    break;
                }
                default:
                    throw new NotImplementedException($"DataType {dataType} not supported");
            }
        }

        /// <summary>
        /// Initializes and returns a vector `Constant` from a given name and float array.
        /// </summary>
        /// <param name="name">The name to use for the constant.</param>
        /// <param name="value">The float array of values.</param>
        public Constant(string name, float[] value)
        {
            this.name = name;
            this.offset = 0;
            this.shape = new TensorShape(value.Length);
            this.length = value.Length;
            this.dataType = DataType.Float;
            weights = new NativeTensorArray(value.Length);
            NativeTensorArray.Copy(value, weights);
        }

        /// <summary>
        /// Initializes and returns a vector `Constant` from a given name and int array.
        /// </summary>
        /// <param name="name">The name to use for the constant.</param>
        /// <param name="value">The int array of values.</param>
        public Constant(string name, int[] value)
        {
            this.name = name;
            this.offset = 0;
            this.shape = new TensorShape(value.Length);
            this.length = value.Length;
            this.dataType = DataType.Int;
            weights = new NativeTensorArray(value.Length);
            NativeTensorArray.Copy(value, weights);
        }

        internal Constant(string name, TensorShape shape, DataType dataType, NativeTensorArray weights)
        {
            this.name = name;
            this.offset = 0;
            this.shape = shape;
            this.length = shape.length;
            this.dataType = dataType;
            this.weights = weights;
        }

        /// <summary>
        /// Returns a string that represents the `Constant`.
        /// </summary>
        /// <returns>A string representation of the `Constant`.</returns>
        public override string ToString()
        {
            return $"Constant{dataType.ToString()} - name: {name}, weights: [{name}, {shape}]";
        }

        /// <summary>
        /// Creates and returns a CPU `Tensor` of the constant.
        /// </summary>
        /// <returns>The created tensor.</returns>
        /// <exception cref="NotImplementedException">Thrown when a given data type is not supported.</exception>
        public Tensor DataSetToTensor()
        {
            switch (dataType)
            {
                case DataType.Float:
                {
                    var array = new float[shape.length];
                    NativeTensorArray.Copy(weights, (int)offset, array, 0, shape.length);
                    return new TensorFloat(shape, array);
                }
                case DataType.Int:
                {
                    var array = new int[shape.length];
                    NativeTensorArray.Copy(weights, (int)offset, array, 0, shape.length);
                    return new TensorInt(shape, array);
                }
                default:
                    throw new NotImplementedException($"DataType {dataType} not supported");
            }
        }

        internal Tensor DataSetToTensorView()
        {
            switch (dataType)
            {
                case DataType.Float:
                {
                    var array = new SharedArrayTensorData(shape, weights, (int)offset);
                    return new TensorFloat(shape, array);
                }
                case DataType.Int:
                {
                    var array = new SharedArrayTensorData(shape, weights, (int)offset);
                    return new TensorInt(shape, array);
                }
                default:
                    throw new NotImplementedException($"DataType {dataType} not supported");
            }
        }

        /// <summary>
        /// Initializes the constant with the shape, dataType and weights from a given `Tensor`.
        /// </summary>
        /// <param name="X">The tensor to use for initialization.</param>
        /// <exception cref="NotImplementedException">Thrown when a given data type is not supported.</exception>
        public void TensorToDataSet(Tensor X)
        {
            this.shape = X.shape;
            this.length = X.shape.length;
            this.dataType = X.dataType;
            weights = new NativeTensorArray(X.shape.length);
            switch (dataType)
            {
                case DataType.Float:
                {
                    NativeTensorArray.Copy(X.ToReadOnlyNativeArray<float>(), 0, weights, (int)offset, shape.length);
                    break;
                }
                case DataType.Int:
                {
                    NativeTensorArray.Copy(X.ToReadOnlyNativeArray<int>(), 0, weights, (int)offset, shape.length);
                    break;
                }
                default:
                    throw new NotImplementedException($"DataType {dataType} not supported");
            }
        }
    }
}
