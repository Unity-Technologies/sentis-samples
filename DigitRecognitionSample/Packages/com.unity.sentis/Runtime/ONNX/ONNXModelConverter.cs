using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using Google.Protobuf;
using Google.Protobuf.Collections;
using Onnx;
using System.Reflection;

[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]
[assembly: InternalsVisibleTo("Unity.Sentis.ONNX")]
[assembly: InternalsVisibleTo("Unity.Sentis.Editor")]

namespace Unity.Sentis.ONNX
{
    /// <summary>
    /// Attribute to define ONNX operator name for custom import.
    /// </summary>
    [AttributeUsage(AttributeTargets.Interface | AttributeTargets.Class | AttributeTargets.Struct)]
    public class OpImportAttribute : Attribute
    {
        /// <summary>
        /// Name of ONNX operator.
        /// </summary>
        public string opType;
        /// <summary>
        /// Instantiates and returns ONNX operator attribute.
        /// </summary>
        /// <param name="opType">Name of ONNX operator.</param>
        public OpImportAttribute(string opType) { this.opType = opType; }
    }

    /// <summary>
    /// An interface that provides methods for importing custom layers.
    /// </summary>
    public interface IOpImporter
    {
        /// <summary>
        /// Member method that implements adding the constants and layer of an ONNX operator node to a model.
        /// </summary>
        /// <param name="net">The current model to add the node to.</param>
        /// <param name="node">The node to import.</param>
        void Import(Model net, OperatorNode node);
    }

    /// <summary>
    /// Represents an ONNX operator for custom import.
    /// </summary>
    public struct OperatorNode
    {
        ONNXNodeWrapper node;
        internal OperatorNode(ONNXNodeWrapper node) { this.node = node; }

        /// <summary>
        /// The name of the ONNX operator.
        /// </summary>
        public string Name { get { return node.Name; } }

        /// <summary>
        /// The outputs of the ONNX operator in order, as a string array.
        /// </summary>
        public string[] Outputs { get { return node.Outputs; } }

        /// <summary>
        /// The inputs of the ONNX operator in order, as a string array.
        /// </summary>
        public string[] Inputs { get { return node.Inputs; } }

        /// <summary>
        /// Whether the ONNX operator has an attribute called `name`.
        /// </summary>
        /// <param name="name">The name of the attribute.</param>
        /// <returns>Whether operator has attribute with name.</returns>
        public bool HasAttribute(string name) { return node.HasAttribute(name); }

        /// <summary>
        /// Gets the float attribute called `name`.
        /// </summary>
        /// <param name="name">The name of the attribute.</param>
        /// <returns>The float value.</returns>
        public float GetFloatAttribute(string name) { return node.GetRequiredFloat(name); }

        /// <summary>
        /// Gets a float list attribute called `name`, as an array of floats.
        /// </summary>
        /// <param name="name">The name of the attribute.</param>
        /// <returns>The float list values as an array.</returns>
        public float[] GetFloatArrayAttribute(string name) { return node.GetRequiredFloatArray(name); }

        /// <summary>
        /// Gets the int attribute called `name`.
        /// </summary>
        /// <param name="name">The name of the attribute.</param>
        /// <returns>The int value.</returns>
        public int GetIntAttribute(string name) { return node.GetRequiredInt(name); }

        /// <summary>
        /// Gets an int list attribute called `name`, as an array of ints.
        /// </summary>
        /// <param name="name">The name of the attribute.</param>
        /// <returns>The int list values as an array.</returns>
        public int[] GetIntArrayAttribute(string name) { return node.GetRequiredIntArray(name); }

        /// <summary>
        /// Gets the string attribute called `name`.
        /// </summary>
        /// <param name="name">The name of the attribute.</param>
        /// <returns>The string value.</returns>
        public string GetStringAttribute(string name) { return node.GetRequiredString(name); }

        /// <summary>
        /// Gets a string list attribute called `name`, as an array of strings.
        /// </summary>
        /// <param name="name">The name of the attribute.</param>
        /// <returns>The string list values as an array.</returns>
        public string[] GetStringArrayAttribute(string name) { return node.GetRequiredStringArray(name); }
    }

    /// <summary>
    /// Represents a converter from an ONNX model to Sentis format.
    /// </summary>
    public class ONNXModelConverter
    {
        // Configuration
        bool m_OptimizeModel;
        string m_DirectoryPath;
        string m_FilePath;

        internal event Action<Dictionary<string, IOpImporter>> CollectOpImporters;

        /// <summary>
        /// Calls the methods in its invocation list when the model is imported.
        /// </summary>
        public static event Action<object, Model> ModelImported;

        void Add(string opType, Action<Model, ONNXNodeWrapper> opImportAction)
        {
            m_NodeImporters.Add(opType, opImportAction);
        }

        /// <summary>
        /// Converts an ONNX model to a Sentis `Model` object.
        /// </summary>
        /// <returns>The converted Sentis model.</returns>
        public Model Convert()
        {
            using var readStream = new FileStream(m_FilePath, FileMode.Open, FileAccess.Read);
            using var inputStream = new CodedInputStream(readStream);

            var onnxModel = new ModelProto();
            onnxModel.MergeFrom(inputStream);

            Model model = null;
            SetupImporter();
            model = ConvertOnnxModel(onnxModel);
            ModelImported?.Invoke(onnxModel, model);

            return model;
        }

        /// <summary>
        /// Initializes and returns an instance of `ONNXModelConverter`.
        /// </summary>
        /// <param name="optimizeModel">Whether to perform optimizations on the model during the import. The default value is `true`.</param>
        /// <param name="filePath">The path of the asset to convert.</param>
        public ONNXModelConverter(bool optimizeModel, string filePath)
        {
            m_OptimizeModel = optimizeModel;

            m_FilePath = filePath;
            m_DirectoryPath = Path.GetDirectoryName(m_FilePath);
        }

        internal void SetupImporter()
        {
            m_NodeImporters.Clear();

            Add("Constant", (net, node) =>
            {
                node.UnsupportedAttribute("sparse_value");
                var constant = ONNXConstantsLoader.LoadConstant(node.ValueAsTensor, m_DirectoryPath);
                constant.name = node.Name;
                net.AddConstant(constant);
            });

            // Layer.Activation
            Add("Celu", (net, node) => { net.AddLayer(new Layers.Celu(node.Name, node.Input0, node.GetOptionalFloat("alpha", 1f))); });
            Add("Elu", (net, node) => { net.AddLayer(new Layers.Elu(node.Name, node.Input0, node.AlphaOptional(1f))); });
            Add("Erf", (net, node) => { net.AddLayer(new Layers.Erf(node.Name, node.Input0)); });
            Add("Gelu", (net, node) =>{ net.AddLayer(new Layers.Gelu(node.Name, node.Input0)); });
            Add("Hardmax", (net, node) =>
            {
                var axis = node.AxisOptional(net.DefaultOpsetVersion > 11 ? -1 : 1);
                net.AddLayer(new Layers.Hardmax(node.Name, node.Input0, axis));
            });
            Add("HardSigmoid", (net, node) => { net.AddLayer(new Layers.HardSigmoid(node.Name, node.Input0, node.AlphaOptional(0.2f), node.BetaOptional(0.5f))); });
            Add("HardSwish", (net, node) => { net.AddLayer(new Layers.HardSwish(node.Name, node.Input0)); });
            Add("LeakyRelu", (net, node) => { net.AddLayer(new Layers.LeakyRelu(node.Name, node.Input0, node.AlphaOptional(0.01f))); });
            Add("PRelu", (net, node) => { net.AddLayer(new Layers.PRelu(node.Name, node.Input0, node.Input1)); });
            Add("Relu", (net, node) => { net.AddLayer(new Layers.Relu(node.Name, node.Input0)); });
            Add("Selu", (net, node) => { net.AddLayer(new Layers.Selu(node.Name, node.Input0, node.AlphaOptional(1.67326f), node.GammaOptional(1.0507f))); });
            Add("Sigmoid", (net, node) => { net.AddLayer(new Layers.Sigmoid(node.Name, node.Input0)); });
            Add("Softplus", (net, node) => { net.AddLayer(new Layers.Softplus(node.Name, node.Input0)); });
            Add("Softsign", (net, node) => { net.AddLayer(new Layers.Softsign(node.Name, node.Input0)); });
            Add("Tanh", (net, node) => { net.AddLayer(new Layers.Tanh(node.Name, node.Input0)); });
            Add("ThresholdedRelu", (net, node) => { net.AddLayer(new Layers.ThresholdedRelu(node.Name, node.Input0, node.GetOptionalFloat("alpha", 1f))); });

            // Layer.ActivationNonLinear
            Add("LogSoftmax", (net, node) =>
            {
                var axis = node.AxisOptional(net.DefaultOpsetVersion > 11 ? -1 : 1);
                net.AddLayer(new Layers.LogSoftmax(node.Name, node.Input0, axis));
            });
            Add("Softmax", (net, node) =>
            {
                var axis = node.AxisOptional(net.DefaultOpsetVersion > 11 ? -1 : 1);
                net.AddLayer(new Layers.Softmax(node.Name, node.Input0, axis));
            });

            // Layer.Convolution
            Add("Conv", (net, node) =>
            {
                // Conv-1, Conv-11

                var autoPad = node.AutoPadMode();
                var kernelShape = node.GetOptionalIntArray("kernel_shape", null);
                var dilations = node.GetOptionalIntArray("dilations", null);
                var group = node.GetOptionalInt("group", 1);
                var pads = node.GetOptionalIntArray("pads", null);
                var strides = node.GetOptionalIntArray("strides", null);

                if (node.InputCount == 2)
                    net.AddLayer(new Layers.Conv(node.Name, node.Input0, node.Input1, group, strides, pads, dilations, autoPad, kernelShape));
                else
                    net.AddLayer(new Layers.Conv(node.Name, node.Input0, node.Input1, node.Input2, group, strides, pads, dilations, autoPad, kernelShape));
            });
            Add("ConvTranspose", (net, node) =>
            {
                // ConvTranspose-1, ConvTranspose-11

                node.UnsupportedAttribute("output_shape", "null");

                var outputPadding = node.GetOptionalIntArray("output_padding", null);
                var autoPad = node.AutoPadMode();
                var kernelShape = node.GetOptionalIntArray("kernel_shape", null);
                node.UnsupportedAttribute("dilations", "null");
                node.UnsupportedAttribute("group", 1);
                var pads = node.GetOptionalIntArray("pads", null);
                var strides = node.GetOptionalIntArray("strides", null);

                if (node.InputCount == 2)
                    net.AddLayer(new Layers.ConvTranspose(node.Name, node.Input0, node.Input1, strides, pads, autoPad, outputPadding, kernelShape));
                else
                    net.AddLayer(new Layers.ConvTranspose(node.Name, node.Input0, node.Input1, node.Input2, strides, pads, autoPad, outputPadding, kernelShape));
            });

            // Layer.Dimension
            Add("Shape", (net, node) =>
            {
                // Shape-1, Shape-13, Shape-15
                var start = node.GetOptionalInt("start", 0);
                var end = node.GetOptionalInt("end", TensorShape.maxRank);
                net.AddLayer(new Layers.Shape(node.Name, node.Input0, start, end));
            });
            Add("Size", (net, node) =>
            {
                // Size-1, Size-13
                net.AddLayer(new Layers.Size(node.Name, node.Input0));
            });

            // Layer.Generator
            Add("ConstantOfShape", (net, node) =>
            {
                UnityEngine.Debug.Assert(node.InputCount > 0);

                if (!node.HasAttribute("value"))
                {
                    net.AddLayer(new Layers.ConstantOfShape(node.Name, node.Input0, 0.0f));
                    return;
                }

                var constant = ONNXConstantsLoader.LoadConstant(node.ValueAsTensor, m_DirectoryPath);
                if (constant.dataType == DataType.Int)
                {
                    var value = constant.weights.Get<int>(0);
                    net.AddLayer(new Layers.ConstantOfShape(node.Name, node.Input0, value));
                }
                else
                {
                    var value = constant.weights.Get<float>(0);
                    net.AddLayer(new Layers.ConstantOfShape(node.Name, node.Input0, value));
                }
                constant.weights.Dispose();
            });
            Add("Range", (net, node) =>
            {
                net.AddLayer(new Layers.Range(node.Name, node.Input0, node.Input1, node.Input2));
            });
            Add("OneHot", (net, node) =>
            {
                // OneHot-9, OneHot-11
                var axis = node.AxisOptional(-1);
                net.AddLayer(new Layers.OneHot(node.Name, node.Input0, node.Input1, node.Input2, axis));
            });

            // Layer.Indexing
            Add("ArgMax", (net, node) =>
            {
                var keepdims = node.GetOptionalInt("keepdims", 1) == 1;
                var selectLastIndex = node.GetOptionalInt("select_last_index", 0) == 1;
                var axis = node.AxisOptional(0);
                net.AddLayer(new Layers.ArgMax(node.Name, node.Input0, axis, keepdims, selectLastIndex));
            });
            Add("ArgMin", (net, node) =>
            {
                var keepdims = node.GetOptionalInt("keepdims", 1) == 1;
                var selectLastIndex = node.GetOptionalInt("select_last_index", 0) == 1;
                var axis = node.AxisOptional(0);
                net.AddLayer(new Layers.ArgMin(node.Name, node.Input0, axis, keepdims, selectLastIndex));
            });
            Add("Gather", (net, node) =>
            {
                var axis = node.AxisOptional(0);
                net.AddLayer(new Layers.Gather(node.Name, node.Input0, node.Input1, axis));
            });
            Add("GatherElements", (net, node) =>
            {
                var axis = node.AxisOptional(0);
                net.AddLayer(new Layers.GatherElements(node.Name, node.Input0, node.Input1, axis));
            });
            Add("GatherND", (net, node) =>
            {
                var batchDims = node.GetOptionalInt("batch_dims", 0);
                net.AddLayer(new Layers.GatherND(node.Name, node.Input0, node.Input1, batchDims));
            });
            Add("NonZero", (net, node) =>
            {
                net.AddLayer(new Layers.NonZero(node.Name, node.Input0));
            });
            Add("Scatter", (net, node) =>
            {
                // Scatter-9 maps to ScatterElements
                var axis = node.AxisOptional(0);
                net.AddLayer(new Layers.ScatterElements(node.Name, node.Input0, node.Input1, node.Input2, axis, Layers.ScatterReductionMode.None));
            });
            Add("ScatterElements", (net, node) =>
            {
                int axis = node.AxisOptional(0);
                Layers.ScatterReductionMode reduction = node.ScatterReductionMode();
                net.AddLayer(new Layers.ScatterElements(node.Name, node.Input0, node.Input1, node.Input2, axis, reduction));
            });
            Add("ScatterND", (net, node) =>
            {
                Layers.ScatterReductionMode reduction = node.ScatterReductionMode();
                net.AddLayer(new Layers.ScatterND(node.Name, node.Input0, node.Input1, node.Input2, reduction));
            });
            Add("TopK", (net, node) =>
            {
                string[] outputs = { node.Outputs[0], node.Outputs[1] };
                var axis = node.AxisOptional(-1);
                var largest = node.GetOptionalInt("largest", 1) == 1;
                var sorted = node.GetOptionalInt("sorted", 1) == 1;
                if (node.HasAttribute("k"))
                {
                    // TopK-1
                    var k = node.GetRequiredInt("k");
                    var kConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_k"), new[] { k });
                    net.AddConstant(kConstant);
                    net.AddLayer(new Layers.TopK(node.Name, node.Input0, kConstant.name, axis, largest, sorted, outputs));
                }
                else
                {
                    // TopK-10, TopK-11
                    net.AddLayer(new Layers.TopK(node.Name, node.Input0, node.Input1, axis, largest, sorted, outputs));
                }
            });

            // Layer.Logical
            Add("And", (net, node) => { net.AddLayer(new Layers.And(node.Name, node.Input0, node.Input1)); });
            Add("Compress", (net, node) =>
            {
                if (node.HasAttribute("axis"))
                    net.AddLayer(new Layers.Compress(node.Name, node.Input0, node.Input1, node.Axis));
                else
                    net.AddLayer(new Layers.Compress(node.Name, node.Input0, node.Input1));
            });
            Add("Equal", (net, node) => { net.AddLayer(new Layers.Equal(node.Name, node.Input0, node.Input1)); });
            Add("Greater", (net, node) => { net.AddLayer(new Layers.Greater(node.Name, node.Input0, node.Input1)); });
            Add("GreaterOrEqual", (net, node) => { net.AddLayer(new Layers.GreaterOrEqual(node.Name, node.Input0, node.Input1)); });
            Add("IsInf", (net, node) =>
            {
                var detectNegative = node.GetOptionalInt("detect_negative", 1) != 0;
                var detectPositive = node.GetOptionalInt("detect_positive", 1) != 0;
                net.AddLayer(new Layers.IsInf(node.Name, node.Input0, detectNegative, detectPositive));
            });
            Add("IsNaN", (net, node) => { net.AddLayer(new Layers.IsNaN(node.Name, node.Input0)); });
            Add("Less", (net, node) => { net.AddLayer(new Layers.Less(node.Name, node.Input0, node.Input1)); });
            Add("LessOrEqual", (net, node) => { net.AddLayer(new Layers.LessOrEqual(node.Name, node.Input0, node.Input1)); });
            Add("Not", (net, node) => { net.AddLayer(new Layers.Not(node.Name, node.Input0)); });
            Add("Or", (net, node) => { net.AddLayer(new Layers.Or(node.Name, node.Input0, node.Input1)); });
            Add("Xor", (net, node) => { net.AddLayer(new Layers.Xor(node.Name, node.Input0, node.Input1)); });
            Add("Where", (net, node) => { net.AddLayer(new Layers.Where(node.Name, node.Input0, node.Input1, node.Input2)); });

            // Layer.Math
            Add("Abs", (net, node) => { net.AddLayer(new Layers.Abs(node.Name, node.Input0)); });
            Add("Add", (net, node) => { net.AddLayer(new Layers.Add(node.Name, node.Input0, node.Input1)); });
            Add("Ceil", (net, node) => { net.AddLayer(new Layers.Ceil(node.Name, node.Input0)); });
            Add("Clip", (net, node) =>
            {
                if (node.HasAttribute("min") || node.HasAttribute("max"))
                {
                    // Clip-1, Clip-6 with at least one attribute from min/max
                    var min = node.GetOptionalFloat("min", float.MinValue);
                    var minConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_min"), new[] { min });
                    net.AddConstant(minConstant);
                    var max = node.GetOptionalFloat("max", float.MaxValue);
                    var maxConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_max"), new[] { max });
                    net.AddConstant(maxConstant);
                    net.AddLayer(new Layers.Clip(node.Name, node.Input0, minConstant.name, maxConstant.name));
                }
                else
                {
                    // Clip-11, Clip-12, Clip-13 or Clip-1, Clip-6 with no min or max
                    var minInput = node.InputCount >= 2 ? node.Inputs[1] : "";
                    var maxInput = node.InputCount >= 3 ? node.Inputs[2] : "";
                    if (string.IsNullOrEmpty(minInput))
                        net.AddLayer(new Layers.Clip(node.Name, node.Input0));
                    else if (string.IsNullOrEmpty(maxInput))
                        net.AddLayer(new Layers.Clip(node.Name, node.Input0, node.Input1));
                    else
                        net.AddLayer(new Layers.Clip(node.Name, node.Input0, node.Input1, node.Input2));
                }
            });
            Add("CumSum", (net, node) =>
            {
                var reverse = node.GetOptionalInt("reverse", 0) == 1;
                var exclusive = node.GetOptionalInt("exclusive", 0) == 1;
                net.AddLayer(new Layers.CumSum(node.Name, node.Input0, node.Input1, reverse, exclusive));
            });
            Add("Div", (net, node) => { net.AddLayer(new Layers.Div(node.Name, node.Input0, node.Input1)); });
            Add("Einsum", (net, node) =>
            {
                net.AddLayer(new Layers.Einsum(node.Name, node.Inputs, node.GetRequiredString("equation")));
            });
            Add("Exp", (net, node) => { net.AddLayer(new Layers.Exp(node.Name, node.Input0)); });
            Add("Floor", (net, node) => { net.AddLayer(new Layers.Floor(node.Name, node.Input0)); });
            Add("Gemm", (net, node) =>
            {
                node.UnsupportedAttribute("alpha", 1.0f);
                node.UnsupportedAttribute("beta", 1.0f);

                var transposeA = node.GetOptionalInt("transA", 0) == 1;
                var transposeB = node.GetOptionalInt("transB", 0) == 1;

                var name = node.Name;
                if (node.InputCount == 3)
                    name += "_Gemm";

                net.AddLayer(new Layers.MatMul2D(name, node.Input0, transposeA, node.Input1, transposeB));

                if (node.InputCount == 3)
                {
                    net.AddLayer(new Layers.Add(node.Name, name, node.Input2));
                }
            });
            Add("Log", (net, node) => { net.AddLayer(new Layers.Log(node.Name, node.Input0)); });
            Add("MatMul", (net, node) =>
            {
                net.AddLayer(new Layers.MatMul(node.Name, node.Input0, node.Input1));
            });
            Add("Max", (net, node) => { net.AddLayer(new Layers.Max(node.Name, node.Inputs)); });
            Add("Mean", (net, node) => { net.AddLayer(new Layers.Mean(node.Name, node.Inputs)); });
            Add("Min", (net, node) => { net.AddLayer(new Layers.Min(node.Name, node.Inputs)); });
            Add("Mod", (net, node) => { net.AddLayer(new Layers.Mod(node.Name, node.Input0, node.Input1, node.GetOptionalInt("fmod", 0) != 0)); });
            Add("Mul", (net, node) => { net.AddLayer(new Layers.Mul(node.Name, node.Input0, node.Input1)); });
            Add("Neg", (net, node) => { net.AddLayer(new Layers.Neg(node.Name, node.Input0)); });
            Add("Pow", (net, node) =>
            {
                // Pow-1, Pow-7, Pow-12, Pow-13
                net.AddLayer(new Layers.Pow(node.Name, node.Input0, node.Input1));
            });
            Add("Reciprocal", (net, node) => { net.AddLayer(new Layers.Reciprocal(node.Name, node.Input0)); });
            Add("Round", (net, node) => { net.AddLayer(new Layers.Round(node.Name, node.Input0)); });
            Add("Shrink", (net, node) => { net.AddLayer(new Layers.Shrink(node.Name, node.Input0, node.GetOptionalFloat("bias", 0f), node.GetOptionalFloat("lambd", 0.5f))); });
            Add("Sign", (net, node) => { net.AddLayer(new Layers.Sign(node.Name, node.Input0)); });
            Add("Sqrt", (net, node) => { net.AddLayer(new Layers.Sqrt(node.Name, node.Input0)); });
            Add("Sub", (net, node) => { net.AddLayer(new Layers.Sub(node.Name, node.Input0, node.Input1)); });
            Add("Sum", (net, node) => { net.AddLayer(new Layers.Sum(node.Name, node.Inputs)); });

            // Layer.Normalization
            Add("BatchNormalization", (net, node) =>
            {
                net.AddLayer(new Layers.BatchNormalization(node.Name, node.Input0, node.Input1, node.Input2, node.Input3, node.Input4, node.EpsilonOptional()));
            });
            Add("InstanceNormalization", (net, node) =>
            {
                net.AddLayer(new Layers.InstanceNormalization(node.Name, node.Input0, node.Input1, node.Input2, node.EpsilonOptional()));
            });
            Add("LayerNormalization", (net, node) =>
            {
                node.UnsupportedAttribute("axis", -1);
                net.AddLayer(new Layers.LayerNormalization(node.Name, node.Input0, node.Input1, node.Input2, node.EpsilonOptional()));
            });
            Add("LRN", (net, node) =>
            {
                var bias = node.GetOptionalFloat("bias", 1.0f);
                var size = node.GetRequiredInt("size");
                net.AddLayer(new Layers.LRN(node.Name, node.Input0, node.AlphaOptional(0.0001f), node.BetaOptional(0.75f), bias, size));
            });

            // Layer.ObjectDetection
            Add("NonMaxSuppression", (net, node) =>
            {
                var centerPointBox = (node.GetOptionalInt("center_point_box", 0) == 0) ? Layers.CenterPointBox.Corners : Layers.CenterPointBox.Center;
                var scoreThreshold = node.InputCount == 5 ? node.Input4 : null;
                var iouThreshold = node.InputCount >= 4 ? node.Input3 : null;
                var maxOutputBoxesPerClass = node.InputCount >= 3 ? node.Input2 : null;
                net.AddLayer(new Layers.NonMaxSuppression(node.Name, node.Input0, node.Input1, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, centerPointBox));
            });
            Add("RoiAlign", (net, node) =>
            {
                Layers.RoiPoolingMode mode = node.RoiPoolingMode();

                int output_height = node.GetOptionalInt("output_height", 1);
                int output_width = node.GetOptionalInt("output_width", 1);
                int sampling_ratio = node.GetOptionalInt("sampling_ratio", 0);
                float spatial_scale = node.GetOptionalFloat("spatial_scale", 1.0f);

                net.AddLayer(new Layers.RoiAlign(node.Name, node.Input0, node.Input1, node.Input2, mode, output_height, output_width, sampling_ratio, spatial_scale));
            });

            // Layer.Pooling
            Add("AveragePool", (net, node) =>
            {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("dilations", new[] { 1, 1 });
                node.UnsupportedAttribute("storage_order", 0);
                node.UnsupportedAttribute("count_include_pad", 0);

                var autopad = node.AutoPadMode();

                var kernelShape = node.GetRequiredIntArray("kernel_shape");
                var pads = node.GetOptionalIntArray("pads", null);
                var strides = node.GetOptionalIntArray("strides", null);

                net.AddLayer(new Layers.AveragePool(node.Name, node.Input0, kernelShape, strides, pads, autopad));
            });
            Add("GlobalAveragePool", (net, node) =>
            {
                net.AddLayer(new Layers.GlobalAveragePool(node.Name, node.Input0));
            });
            Add("GlobalMaxPool", (net, node) =>
            {
                net.AddLayer(new Layers.GlobalMaxPool(node.Name, node.Input0));
            });
            Add("MaxPool", (net, node) =>
            {
                node.UnsupportedAttribute("ceil_mode", 0);
                node.UnsupportedAttribute("dilations", new[] { 1, 1 });
                node.UnsupportedAttribute("storage_order", 0);

                var autopad = node.AutoPadMode();

                var kernelShape = node.GetRequiredIntArray("kernel_shape");
                var pads = node.GetOptionalIntArray("pads", null);
                var strides = node.GetOptionalIntArray("strides", null);

                net.AddLayer(new Layers.MaxPool(node.Name, node.Input0, kernelShape, strides, pads, autopad));
            });

            // Layer.Random
            Add("Bernoulli", (net, node) =>
            {
                var dataType = node.GetDataType(defaultValue: DataType.Float);
                net.AddLayer(new Layers.Bernoulli(node.Name, node.Input0, dataType, node.Seed));
            });
            Add("Multinomial", (net, node) =>
            {
                node.IgnoredAttribute("dtype", "dtype can only be int32 or int64 which both map to TensorInt");
                var samples = node.GetOptionalInt("sample_size", 1);
                net.AddLayer(new Layers.Multinomial(node.Name, node.Input0, samples, node.Seed));
            });
            Add("RandomNormal", (net, node) =>
            {
                var mean = node.GetOptionalFloat("mean", 0.0f);
                var scale = node.GetOptionalFloat("scale", 1.0f);
                net.AddLayer(new Layers.RandomNormal(node.Name, node.Shape, mean, scale, node.Seed));
            });
            Add("RandomNormalLike", (net, node) =>
            {
                var mean = node.GetOptionalFloat("mean", 0.0f);
                var scale = node.GetOptionalFloat("scale", 1.0f);
                net.AddLayer(new Layers.RandomNormalLike(node.Name, node.Input0, mean, scale, node.Seed));
            });
            Add("RandomUniform", (net, node) =>
            {
                var low = node.GetOptionalFloat("low", 0.0f);
                var high = node.GetOptionalFloat("high", 1.0f);
                net.AddLayer(new Layers.RandomUniform(node.Name, node.Shape, low, high, node.Seed));
            });
            Add("RandomUniformLike", (net, node) =>
            {
                var low = node.GetOptionalFloat("low", 0.0f);
                var high = node.GetOptionalFloat("high", 1.0f);
                net.AddLayer(new Layers.RandomUniformLike(node.Name, node.Input0, low, high, node.Seed));
            });

            // Layer.Recurrent
            Add("LSTM", (net, node) =>
            {
                var hiddenSize = node.GetRequiredInt("hidden_size");
                var direction = node.Direction();
                var activations = node.Activations();
                var activationAlpha = node.GetOptionalFloatArray("activation_alpha", null);
                var activationBeta = node.GetOptionalFloatArray("activation_beta", null);
                var clip = node.GetOptionalFloat("clip", float.MaxValue);
                var inputForget = node.GetOptionalInt("input_forget", 0) != 0;
                var layout = node.Layout();

                net.AddLayer(new Layers.LSTM(node.Name, node.Inputs, node.Outputs, hiddenSize, direction, activations, activationAlpha, activationBeta, clip, inputForget, layout));
            });

            // Layer.Reduction
            Add("ReduceL1", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.name };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceL1(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceL2", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.name };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceL2(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceLogSum", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.name };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceLogSum(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceLogSumExp", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.name };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceLogSumExp(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceMax", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.name };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceMax(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceMean", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.name };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceMean(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceMin", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.name };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceMin(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceProd", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.name };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceProd(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceSum", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.name };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceSum(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });
            Add("ReduceSumSquare", (net, node) =>
            {
                var inputs = node.Inputs;
                if (node.HasAttribute("axes"))
                {
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    inputs = new[] { node.Input0, axesConstant.name };
                }

                var keepDims = node.GetOptionalInt("keepdims", 1) == 1;
                var noopWithEmptyAxes = node.GetOptionalInt("noop_with_empty_axes", 0) == 1;
                net.AddLayer(new Layers.ReduceSumSquare(node.Name, inputs, keepDims, noopWithEmptyAxes));
            });

            // Layer.Transformation
            Add("Cast", (net, node) =>
            {
                var toOnnxType = (TensorProto.Types.DataType)node.GetRequiredInt("to");
                var toDataType = ONNXNodeWrapper.DataTypeFromOnnxDataType(toOnnxType, OnUnsupported: () =>
                {
                    Warn(net, node, Model.WarningType.Error, $"Unsupported tensor dataType: {toOnnxType}.");
                    Debug.LogError(net.Warnings.Last().Message);
                });
                net.AddLayer(new Layers.Cast(node.Name, node.Input0, toDataType));
            });
            Add("CastLike", (net, node) =>
            {
                net.AddLayer(new Layers.CastLike(node.Name, node.Input0, node.Input1));
            });
            Add("Concat", (net, node) =>
            {
                net.AddLayer(new Layers.Concat(node.Name, node.Inputs, node.Axis));
            });
            Add("DepthToSpace", (net, node) =>
            {
                var modeType = node.ModeOptional("DCR");
                var mode = modeType == "DCR" ? Layers.DepthToSpaceMode.DepthColumnRow : Layers.DepthToSpaceMode.ColumnRowDepth;
                net.AddLayer(new Layers.DepthToSpace(node.Name, node.Input0, node.BlockSize, mode));
            });
            Add("Expand", (net, node) =>
            {
                // Expand-8, Expand-13
                net.AddLayer(new Layers.Expand(node.Name, node.Input0, node.Input1));
            });
            Add("Flatten", (net, node) =>
            {
                var axis = node.AxisOptional(1);
                net.AddLayer(new Layers.Flatten(node.Name, node.Input0, axis));
            });
            Add("Dropout", (net, node) => { net.AddLayer(new Layers.Identity(node.Name, node.Input0)); });
            Add("Identity", (net, node) => { net.AddLayer(new Layers.Identity(node.Name, node.Input0)); });
            Add("Pad", (net, node) =>
            {
                if (node.InputCount > 3)
                    node.Warn($"<b>Pad:</b> Unsupported input `<b>axes</b>`. Value will be ignored and defaulted to [0, 1, ..., input_rank-1].", Model.WarningType.Warning);
                var mode = node.PadMode();
                if (node.InputCount == 1)
                {
                    // Pad-1 or Pad-2
                    var pads = node.GetRequiredIntArray(node.HasAttribute("pads") ? "pads" : "paddings");
                    var padsConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_pads"), pads);
                    net.AddConstant(padsConstant);
                    var value = node.GetOptionalFloat("value", 0f);
                    var valueConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_value"), new[] { value });
                    net.AddConstant(valueConstant);
                    net.AddLayer(new Layers.Pad(node.Name, node.Input0, padsConstant.name, valueConstant.name, mode));
                }
                else if (node.InputCount == 2 || string.IsNullOrEmpty(node.Inputs[2]))
                {
                    // Pad-11, Pad-13, Pad-18 no constant value
                    net.AddLayer(new Layers.Pad(node.Name, node.Input0, node.Input1, mode));
                }
                else
                {
                    // Pad-11, Pad-13, Pad-18 with constant value
                    net.AddLayer(new Layers.Pad(node.Name, node.Input0, node.Input1, node.Input2, mode));
                }
            });
            Add("Reshape", (net, node) =>
            {
                if (node.HasAttribute("shape"))
                {
                    // Reshape-1, Reshape-5
                    var shape = node.GetRequiredIntArray("shape");
                    var shapeConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_shape"), shape);
                    net.AddConstant(shapeConstant);
                    net.AddLayer(new Layers.Reshape(node.Name, node.Input0, shapeConstant.name));
                }
                else
                {
                    // Reshape-13, Reshape-14
                    var allowZero = node.GetOptionalInt("allowzero", 0) != 0;
                    net.AddLayer(new Layers.Reshape(node.Name, node.Input0, node.Input1, allowZero));
                }
            });
            Add("Resize", (net, node) =>
            {
                node.UnsupportedAttribute("cubic_coeff_a", -0.75f);
                node.UnsupportedAttribute("exclude_outside", 0);
                node.UnsupportedAttribute("extrapolation_value", 0);
                var coordinateTransformMode = node.CoordinateTransformMode();
                var mode = node.InterpolationMode();
                var nearestMode = node.NearestMode();

                if (node.InputCount == 2)
                {
                    // Resize-10
                    net.AddLayer(new Layers.Resize(node.Name, node.Input0, node.Input1, Layers.ScaleMode.Scales, mode, coordinateTransformMode, nearestMode));
                }
                else if (node.InputCount == 3)
                {
                    // Resize-11, Resize-13 with scales
                    net.AddLayer(new Layers.Resize(node.Name, node.Input0, node.Input2, Layers.ScaleMode.Scales, mode, coordinateTransformMode, nearestMode));
                }
                else if (node.InputCount == 4)
                {
                    // Resize-11, Resize-13 with sizes
                    net.AddLayer(new Layers.Resize(node.Name, node.Input0, node.Input3, Layers.ScaleMode.Sizes, mode, coordinateTransformMode, nearestMode));
                }
            });
            Add("Slice", (net, node) =>
            {
                if (node.HasAttribute("starts"))
                {
                    // Slice-1
                    var starts = node.GetRequiredIntArray("starts");
                    var startsConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_starts"), starts);
                    net.AddConstant(startsConstant);
                    var ends = node.GetRequiredIntArray("ends");
                    var endsConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_ends"), ends);
                    net.AddConstant(endsConstant);
                    if (node.HasAttribute("axes"))
                    {
                        var axes = node.GetRequiredIntArray("axes");
                        var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                        net.AddConstant(axesConstant);
                        net.AddLayer(new Layers.Slice(node.Name, node.Input0, startsConstant.name, endsConstant.name, axesConstant.name));
                    }
                    else
                    {
                        net.AddLayer(new Layers.Slice(node.Name, node.Input0, startsConstant.name, endsConstant.name));
                    }
                }
                else
                {
                    // Slice-10, Slice-11, Slice-13
                    if (node.InputCount == 3)
                        net.AddLayer(new Layers.Slice(node.Name, node.Input0, node.Input1, node.Input2));
                    else if (node.InputCount == 4)
                        net.AddLayer(new Layers.Slice(node.Name, node.Input0, node.Input1, node.Input2, node.Input3));
                    else if (node.InputCount == 5)
                        net.AddLayer(new Layers.Slice(node.Name, node.Input0, node.Input1, node.Input2, node.Input3, node.Input4));
                }
            });
            Add("SpaceToDepth", (net, node) =>
            {
                net.AddLayer(new Layers.SpaceToDepth(node.Name, node.Input0, node.BlockSize));
            });
            Add("Split", (net, node) =>
            {
                var axis = node.AxisOptional(0);
                if (node.HasAttribute("num_outputs"))
                {
                    // Split-18 with "num_outputs" attribute
                    net.AddLayer(new Layers.Split(node.Name, node.Input0, node.Outputs, axis, node.GetRequiredInt("num_outputs")));
                }
                else if (node.HasAttribute("split"))
                {
                    // Split-1, Split-2, Split-11 with "split" attribute
                    var split = node.GetRequiredIntArray("split");
                    var splitConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_split"), split);
                    net.AddConstant(splitConstant);
                    net.AddLayer(new Layers.Split(node.Name, node.Input0, splitConstant.name, node.Outputs, axis));
                }
                else if (node.InputCount == 2)
                {
                    // Split-1, Split-13, Split-18 with "split" input
                    net.AddLayer(new Layers.Split(node.Name, node.Input0, node.Input1, node.Outputs, axis));
                }
                else
                {
                    // Split-1, Split-2, Split-11, Split-13, Split-18 with no given "split" or "num_outputs"
                    net.AddLayer(new Layers.Split(node.Name, node.Input0, node.Outputs, axis, node.Outputs.Length));
                }
            });
            Add("Squeeze", (net, node) =>
            {
                if (node.HasAttribute("axes"))
                {
                    // Squeeze-1, Squeeze-11 with given axes
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    net.AddLayer(new Layers.Squeeze(node.Name, node.Input0, axesConstant.name));
                }
                else
                {
                    // Squeeze-13 or Squeeze-1, Squeeze-11 without given axes
                    if (node.InputCount == 2)
                        net.AddLayer(new Layers.Squeeze(node.Name, node.Input0, node.Input1));
                    else
                        net.AddLayer(new Layers.Squeeze(node.Name, node.Input0));
                }
            });
            Add("Tile", (net, node) =>
            {
                net.AddLayer(new Layers.Tile(node.Name, node.Input0, node.Input1));
            });
            Add("Transpose", (net, node) =>
            {
                var permutations = node.GetOptionalIntArray("perm", null);
                net.AddLayer(new Layers.Transpose(node.Name, node.Input0, permutations));
            });
            Add("Trilu", (net, node) =>
            {
                var upper = node.GetOptionalInt("upper", 1);

                if (node.InputCount == 1)
                    net.AddLayer(new Layers.Trilu(node.Name, node.Input0, (Layers.TriluMode)upper));
                else
                    net.AddLayer(new Layers.Trilu(node.Name, node.Input0, node.Input1, (Layers.TriluMode)upper));
            });
            Add("Upsample", (net, node) =>
            {
                var coordinateTransformMode = Layers.CoordTransformMode.Asymmetric;
                var mode = node.InterpolationMode();
                var nearestMode = Layers.NearestMode.Floor;
                if (node.HasAttribute("scales"))
                {
                    // Upsample-7
                    var scales = node.GetRequiredFloatArray("scales");
                    var scalesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_scales"), scales);
                    net.AddConstant(scalesConstant);
                    net.AddLayer(new Layers.Resize(node.Name, node.Input0, scalesConstant.name, Layers.ScaleMode.Scales, mode, coordinateTransformMode, nearestMode));
                }
                else
                {
                    // Upsample-9
                    net.AddLayer(new Layers.Resize(node.Name, node.Input0, node.Input1, Layers.ScaleMode.Scales, mode, coordinateTransformMode, nearestMode));
                }
            });
            Add("Unsqueeze", (net, node) =>
            {
                if (node.HasAttribute("axes"))
                {
                    // Unsqueeze-1, Unsqueeze-11
                    var axes = node.GetRequiredIntArray("axes");
                    var axesConstant = new Layers.Constant(net.GetUniqueName(node.Name + "_axes"), axes);
                    net.AddConstant(axesConstant);
                    net.AddLayer(new Layers.Unsqueeze(node.Name, node.Input0, axesConstant.name));
                }
                else
                {
                    // Unsqueeze-13
                    net.AddLayer(new Layers.Unsqueeze(node.Name, node.Input0, node.Input1));
                }
            });

            // Layer.Trigonometric
            Add("Acos", (net, node) => { net.AddLayer(new Layers.Acos(node.Name, node.Input0)); });
            Add("Acosh", (net, node) => { net.AddLayer(new Layers.Acosh(node.Name, node.Input0)); });
            Add("Asin", (net, node) => { net.AddLayer(new Layers.Asin(node.Name, node.Input0)); });
            Add("Asinh", (net, node) => { net.AddLayer(new Layers.Asinh(node.Name, node.Input0)); });
            Add("Atan", (net, node) => { net.AddLayer(new Layers.Atan(node.Name, node.Input0)); });
            Add("Atanh", (net, node) => { net.AddLayer(new Layers.Atanh(node.Name, node.Input0)); });
            Add("Cos", (net, node) => { net.AddLayer(new Layers.Cos(node.Name, node.Input0)); });
            Add("Cosh", (net, node) => { net.AddLayer(new Layers.Cosh(node.Name, node.Input0)); });
            Add("Sin", (net, node) => { net.AddLayer(new Layers.Sin(node.Name, node.Input0)); });
            Add("Sinh", (net, node) => { net.AddLayer(new Layers.Sinh(node.Name, node.Input0)); });
            Add("Tan", (net, node) => { net.AddLayer(new Layers.Tan(node.Name, node.Input0)); });

            // Non standard ONNX
            Add("Swish", (net, node) => { net.AddLayer(new Layers.Swish(node.Name, node.Input0)); });
            Add("ImageScaler", (net, node) =>
            {
                var attrBias = node.Bias;
                var maxElements = attrBias.Length;
                var attrScale = Enumerable.Repeat(node.GetOptionalFloat("scale", 1.0f), maxElements).ToArray();

                using var scale = new TensorFloat(new TensorShape(maxElements), attrScale);
                using var bias = new TensorFloat(new TensorShape(maxElements), attrBias);

                var scaleConstantName = net.GetUniqueName($"{node.Name}_Scale");
                net.AddConstant(new Layers.Constant(scaleConstantName, scale));
                var biasConstantName = net.GetUniqueName($"{node.Name}_Bias");
                net.AddConstant(new Layers.Constant(biasConstantName, bias));
                net.AddLayer(new Layers.ScaleBias(node.Name, node.Input0, scaleConstantName, biasConstantName));
            });
        }

        internal static readonly Dictionary<string, Action<Model, ONNXNodeWrapper>> m_NodeImporters =
            new Dictionary<string, Action<Model, ONNXNodeWrapper>>();

        // NOTE: It's questionable whether we should be doing this since the ONNX specification requires the graph to be
        // topologically sorted, but at least one network encountered that was exported from keras2onnx v1.7.0 produced
        // an incorrectly sorted graph. related example: https://github.com/onnx/keras-onnx/issues/184
        static List<NodeProto> SortTopologically(ModelProto onnxModel)
        {
            GraphProto onnxGraph = onnxModel.Graph;
            HashSet<string> encounteredNodes = new HashSet<string>();
            foreach (var i in onnxGraph.Input)
                encounteredNodes.Add(i.Name);
            foreach (var i in onnxGraph.Initializer)
                encounteredNodes.Add(i.Name);

            var sortedGraph = new List<NodeProto>();
            bool graphInSortedOrder = true;
            foreach (NodeProto node in onnxGraph.Node)
            {
                foreach (var input in node.Input)
                    graphInSortedOrder &= encounteredNodes.Contains(input);

                if (!graphInSortedOrder)
                    break;

                foreach (var output in node.Output)
                    encounteredNodes.Add(output);
                sortedGraph.Add(node);
            }

            if (graphInSortedOrder)
                return sortedGraph;

            sortedGraph.Clear();
            var nodesToSort = new Queue<NodeProto>();
            foreach (NodeProto node in onnxGraph.Node)
            {
                nodesToSort.Enqueue(node);
            }

            var requeueNodes = new Queue<NodeProto>();
            while (nodesToSort.Count > 0)
            {
                NodeProto node = nodesToSort.Dequeue();

                var allInputsExist = true;
                foreach (string input in node.Input)
                {
                    if (string.IsNullOrEmpty(input))
                        continue;

                    if (!sortedGraph.Exists(n => n.Output.Any(o => o == input))
                        && !onnxGraph.Input.Any(i => i.Name == input)
                        && !onnxGraph.Initializer.Any(i => i.Name == input))
                    {
                        allInputsExist = false;
                        break;
                    }
                }

                if (!allInputsExist)
                {
                    if (nodesToSort.Count != 0)
                    {
                        // Mark for re-processing again when (potentially) all inputs have been processed
                        // We use a separate list, so we don't continually spin on nodes that are missing inputs
                        if (!requeueNodes.Contains(node))
                            requeueNodes.Enqueue(node);
                        continue;
                    }

                    // Something must've gone wrong
                    throw new OnnxImportException($"Missing inputs to node {node.Name}, but there are no nodes to process.");
                }

                if (!sortedGraph.Contains(node))
                    sortedGraph.Add(node);

                // Now that we have at least processed a single new node, let's requeue
                while (requeueNodes.Count > 0)
                    nodesToSort.Enqueue(requeueNodes.Dequeue());
            }

            return sortedGraph;
        }

        Model ConvertOnnxModel(ModelProto onnxModel)
        {
            var model = new Model();

            // Parse producer meta data
            foreach (var opsetSetIdProto in onnxModel.OpsetImport)
            {
                model.OpsetDescriptions.Add(new Model.OpsetDescription
                {
                    domain = opsetSetIdProto.Domain,
                    version = opsetSetIdProto.Version
                });
                if (string.IsNullOrEmpty(opsetSetIdProto.Domain))
                    model.DefaultOpsetVersion = opsetSetIdProto.Version;
            }
            model.ProducerName = onnxModel.ProducerName;
            if (!String.IsNullOrEmpty(onnxModel.ProducerVersion))
                model.ProducerName += $" v{onnxModel.ProducerVersion}";
            model.IrSource = "ONNX";
            model.IrVersion = onnxModel.IrVersion;

            // Import any (optional) metadata properties
            RepeatedField<StringStringEntryProto> metadataProps = onnxModel.MetadataProps;
            Dictionary<string, string> metadata = model.Metadata;
            for (int p = 0; p < metadataProps.Count; p++)
            {
                StringStringEntryProto prop = metadataProps[p];
                metadata.Add(prop.Key, prop.Value);
            }

            // Convert graph inputs & outputs
            var initializersByName = onnxModel.Graph.Initializer.ToDictionary(i => i.Name, i => true);
            var namedDims = new List<string>();
            foreach (var input in onnxModel.Graph.Input)
            {
                // skip input tensors that have initializer data, they are constant tensors not global inputs
                // also skip nodes that should be trimmed
                if (initializersByName.ContainsKey(input.Name))
                    continue;

                var onnxShape = input.Type.TensorType.Shape;
                var inputShape = SymbolicTensorShape.UnknownOfRank(onnxShape.Dim.Count);

                for (var i = 0; i < inputShape.rank; i++)
                {
                    var dim = onnxShape.Dim[i];
                    switch (dim.ValueCase)
                    {
                        case TensorShapeProto.Types.Dimension.ValueOneofCase.None:
                            inputShape[i] = SymbolicTensorDim.Unknown;
                            break;
                        case TensorShapeProto.Types.Dimension.ValueOneofCase.DimParam:
                            var index = namedDims.IndexOf(dim.DimParam);
                            if (index < 0)
                            {
                                index = namedDims.Count;
                                namedDims.Add(dim.DimParam);
                            }
                            inputShape[i] = new SymbolicTensorDim((char)index);
                            break;
                        case TensorShapeProto.Types.Dimension.ValueOneofCase.DimValue:
                            if (dim.DimValue < 0)
                                Warn(model, "Input", Model.WarningType.Warning, "ONNX: Tensor shape has negative index, treating as unknown dimension");
                            else
                                inputShape[i] = new SymbolicTensorDim(dim.DimValue > int.MaxValue ? int.MaxValue : (int)dim.DimValue);
                            break;
                        default:
                            throw new ArgumentOutOfRangeException();
                    }
                }

                var inputDataType = ONNXNodeWrapper.DataTypeFromOnnxDataType((TensorProto.Types.DataType)input.Type.TensorType.ElemType);

                model.AddInput(input.Name, inputDataType, inputShape);
            }

            if (namedDims.Count > 0)
            {
                // set the named inputs mapping to the model
                var remapNamedDims = new Dictionary<char, string>();
                for (var i = 0; i < namedDims.Count; i++)
                {
                    remapNamedDims[(char)i] = namedDims[i];
                }
                model.SetRemapNamedDims(remapNamedDims);
            }

            foreach (ValueInfoProto o in onnxModel.Graph.Output)
                model.AddOutput(o.Name);

            var weightsStream = new Dictionary<string, FileStream>();
            // Read constants from initializer list
            foreach (TensorProto initializer in onnxModel.Graph.Initializer)
            {
                if (initializer.DataLocation == TensorProto.Types.DataLocation.External)
                {
                    string name = initializer.ExternalData.Single(x => x.Key == "location").Value;
                    if (!weightsStream.ContainsKey(name))
                    {
                        string filePath = Path.Combine(m_DirectoryPath, name);
                        if (File.Exists(filePath))
                            weightsStream.Add(name, File.OpenRead(Path.Combine(m_DirectoryPath, name)));
                        else
                        {
                            Warn(model, name, Model.WarningType.Error, $"External Weights file not found! Expecting: {filePath}");
                            return model;
                        }
                    }
                    var stream = weightsStream[name];
                    var constant = ONNXConstantsLoader.LoadConstant(initializer, stream);
                    model.AddConstant(constant);
                }
                else
                {
                    var constant = ONNXConstantsLoader.LoadConstant(initializer);
                    model.AddConstant(constant);
                }
            }
            foreach (var stream in weightsStream.Values)
                stream.Dispose();

            // Nodes are supposed to be sorted, but this isn't always the case
            var sortedGraph = SortTopologically(onnxModel);

            Dictionary<string, IOpImporter> customOps = new Dictionary<string, IOpImporter>();
            CollectOpImporters?.Invoke(customOps);

            // Convert graph nodes
            foreach (NodeProto onnxNode in sortedGraph)
            {
                var node = new ONNXNodeWrapper(onnxNode, model.Warnings);
                var nodeId = node.Name;
                var opType = node.OperatorType;

                if (customOps.ContainsKey(opType))
                {
                    OperatorNode opNode = new OperatorNode(node);
                    customOps[opType].Import(model, opNode);
                    continue;
                }

                if (!m_NodeImporters.ContainsKey(opType))
                {
                    Warn(model, nodeId, Model.WarningType.Error, $"{opType} not supported");
                    Debug.LogError(model.Warnings.Last().Message);
                    continue;
                }

                try
                {
                    m_NodeImporters[opType](model, node);
                }
                catch (Exception e)
                {
                    Warn(model, nodeId, Model.WarningType.Error, e.Message);
                    Debug.LogError(model.Warnings.Last().Message);
                }
            }

            // strip :0 at the end of string name for TF import
            model = TrimTensorflowNames(model);

            // validate imported model
            if (!model.Warnings.Any(w => w.MessageSeverity == Model.WarningType.Error))
            {
                model = ModelValidator.ValidateModel(model);
            }
            if (!model.Warnings.Any(w => w.MessageSeverity == Model.WarningType.Error))
            {
                if (m_OptimizeModel)
                    ModelOptimizer.OptimizeModel(ref model);
                ModelOptimizer.RunCPUFallbackPass(ref model);
            }

            return model;
        }

        static Model TrimTensorflowNames(Model model)
        {
            model.inputs   = model.inputs.Select(i   => {
                i.name = TrimTensorflowName(i.name);
                return i;
            }).ToList();

            model.outputs  = model.outputs.Select(o  => {
                return TrimTensorflowName(o);
            }).ToList();

            model.constants = model.constants.Select(c => {
                c.name = TrimTensorflowName(c.name);
                return c;
            }).ToList();

            model.layers   = model.layers.Select(l   => {
                l.name = TrimTensorflowName(l.name);
                for(int i = 0; i < l.inputs.Length; i++)
                    l.inputs[i] = TrimTensorflowName(l.inputs[i]);
                if (l.outputs != null)
                {
                    for (int i = 0; i < l.outputs.Length; i++)
                        l.outputs[i] = TrimTensorflowName(l.outputs[i]);
                }
                return l;
            }).ToList();

            return model;
        }

        static string TrimTensorflowName(string name)
        {
            if (name.EndsWith(":0"))
                return name.Remove(name.Length-2);
            return name;
        }

        // Logging helpers
        static void Warn(Model model, ONNXNodeWrapper node, Model.WarningType severity, string message)
        {
            Warn(model, node.Name, severity, message);
        }

        static void Warn(Model model, string layerName, Model.WarningType severity, string message)
        {
            model.Warnings.Add(new Model.ImporterWarning(layerName, severity, message));
        }
    }

    /// <summary>
    /// Represents an exception during the import of an ONNX model.
    /// </summary>
    public class OnnxImportException : Exception
    {
        /// <summary>
        /// Initializes and returns an instance of `OnnxImportException`.
        /// </summary>
        /// <param name="message">message</param>
        public OnnxImportException(string message) : base(message) { }
    }

    /// <summary>
    /// Represents an exception during the import of a ONNX layer.
    /// </summary>
    public class OnnxLayerImportException : Exception
    {
        /// <summary>
        /// Initializes and returns an instance of `ONNXLayerImportException`.
        /// </summary>
        /// <param name="message">message</param>
        public OnnxLayerImportException(string message) : base(message) { }
    }
}
