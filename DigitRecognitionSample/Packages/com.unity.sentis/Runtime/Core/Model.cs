using System;
using System.Linq; // Select
using System.Collections.Generic;
using System.Text;

namespace Unity.Sentis {

/// <summary>
/// Represents a Sentis neural network.
/// </summary>
public class Model
{
    /// <summary>
    /// The version of the model. The value increments each time the data structure changes.
    /// </summary>
    public const int Version = 29;
    internal const int WeightsAlignment = 16;

    /// <summary>
    /// Represents an input to a model.
    /// </summary>
    [Serializable]
    public struct Input
    {
        /// <summary>
        /// The name of the input.
        /// </summary>
        public string name;

        /// <summary>
        /// The data type of the input data.
        /// </summary>
        public DataType dataType;

        /// <summary>
        /// The shape of the input, as `SymbolicTensorShape`.
        /// </summary>
        public SymbolicTensorShape shape;
    }

    /// <summary>
    /// The inputs of the model.
    /// </summary>
    public List<Input> inputs = new List<Input>();

    /// <summary>
    /// The outputs of the model.
    /// </summary>
    public List<string> outputs = new List<string>();

    /// <summary>
    /// The layers of the model.
    /// </summary>
    public List<Layers.Layer> layers = new List<Layers.Layer>();

    /// <summary>
    /// The constants of the model.
    /// </summary>
    public List<Layers.Constant> constants = new List<Layers.Constant>();

    /// <summary>
    /// The metadata of the model, as a string.
    /// </summary>
    public string IrSource = "Script";

    /// <summary>
    /// The metadata of the ONNX model, as a string.
    /// </summary>
    public long IrVersion;

    /// <summary>
    /// The producer of the model, as a string.
    /// </summary>
    public string ProducerName = "Script";

    /// <summary>
    /// The opset version number of the ONNX model, for the default domain.
    /// </summary>
    public long DefaultOpsetVersion;

    /// <summary>
    /// The opsets of the ONNX model.
    /// </summary>
    public List<OpsetDescription> OpsetDescriptions = new List<OpsetDescription>();

    /// <summary>
    /// Represents an ONNX opset domain and version number.
    /// </summary>
    public struct OpsetDescription
    {
        /// <summary>
        /// The opset domain of the ONNX model.
        /// </summary>
        public string domain;

        /// <summary>
        /// The opset version number of the ONNX model.
        /// </summary>
        public long version;
    }

    /// <summary>
    /// The warnings from the model importer.
    /// </summary>
    public List<ImporterWarning> Warnings { get; } = new List<ImporterWarning>();

    /// <summary>
    /// Represents types of warning from the model importer.
    /// </summary>
    public enum WarningType
    {
        /// <summary>
        /// No error.
        /// </summary>
        None = 0,

        /// <summary>
        /// Information. Execution should run without errors.
        /// </summary>
        Info = 1,

        /// <summary>
        /// Warning. Execution should run, but may have issues with precision or speed.
        /// </summary>
        Warning = 2,

        /// <summary>
        /// Error. Execution won't run.
        /// </summary>
        Error = 3
    }

    /// <summary>
    /// Represents the data structure for a warning from the model importer.
    /// </summary>
    public class ImporterWarning
    {
        /// <summary>
        /// A message.
        /// </summary>
        public string Message { get; }

        /// <summary>
        /// A layer name.
        /// </summary>
        public string LayerName { get; }

        /// <summary>
        /// The severity of a warning.
        /// </summary>
        public WarningType MessageSeverity { get; }

        /// <summary>
        /// Initializes and returns an instance of `ImporterWarning`.
        /// </summary>
        /// <param name="layer">The name of the layer where the warning originates</param>
        /// <param name="severity">The severity of the warning as a `WarningType`</param>
        /// <param name="msg">The message text of the warning</param>
        public ImporterWarning(string layer, WarningType severity, string msg)
        {
            Message = msg;
            MessageSeverity = severity;
            LayerName = layer;
        }
    }

    /// <summary>
    /// The metadata properties associated with the model.
    /// </summary>
    public Dictionary<string, string> Metadata { get; private set; } = new Dictionary<string, string>();

    /// <summary>
    /// dim param mapping from char to string as dictionary
    /// </summary>
    internal Dictionary<char, string> RemapNamedDims = new Dictionary<char, string>();

    /// <summary>
    /// stores which layers should fallback to CPU for execution
    /// </summary>
    internal HashSet<string> LayerCPUFallback = new HashSet<string>();

    /// <summary>
    /// Returns a shallow copy of the model.
    /// </summary>
    /// <returns>The shallow copy of the model</returns>
    public Model ShallowCopy()
    {
        var model = new Model();
        model.constants.AddRange(constants);
        model.inputs.AddRange(inputs);
        model.outputs.AddRange(outputs);
        model.layers.AddRange(layers);

        model.IrSource = IrSource;
        model.IrVersion = IrVersion;
        model.ProducerName = ProducerName;
        model.DefaultOpsetVersion = DefaultOpsetVersion;
        model.OpsetDescriptions.AddRange(OpsetDescriptions);
        model.Warnings.AddRange(Warnings);
        model.Metadata = new Dictionary<string, string>(Metadata);

        model.RemapNamedDims = new Dictionary<char, string>(RemapNamedDims);

        return model;
    }

    /// <summary>
    /// Returns a string that represents the `Model`.
    /// </summary>
    /// <returns>String representation of model.</returns>
    public override string ToString()
    {
        // weights are not loaded for UI, recompute size
        var totalUniqueWeights = 0;
        return $"inputs: [{string.Join(", ", inputs.Select(i => $"{i.name} {i.shape} [{i.dataType}]"))}], " +
            $"outputs: [{string.Join(", ", outputs)}] " +
            $"\n{layers.Count} layers, {totalUniqueWeights:n0} weights: \n{string.Join("\n", layers.Select(i => $"{i.GetType()} ({i})"))}";
    }

    /// <summary>
    /// Given a symbolic tensor shape return a pretty print string
    /// with the original names of the named param dims
    /// </summary>
    internal string GetSymbolicTensorShapeAsString(SymbolicTensorShape shape)
    {
        if (!shape.hasRank)
            return "Unknown";
        var sb = new StringBuilder();
        sb.Append("(");
        for (var i = 0; i < shape.rank; i++)
        {
            if (i != 0)
                sb.Append(", ");
            var dim = shape[i];

            if (dim.isUnknown)
                sb.Append("?");
            else if (dim.isValue)
                sb.Append(dim.value.ToString());
            else if (RemapNamedDims.TryGetValue(dim.param, out var stringParam))
                sb.Append(stringParam);
            else
                sb.Append(dim.param.ToString());
        }

        sb.Append(")");
        return sb.ToString();
    }

    /// <summary>
    /// Returns a string name not yet used in the model inputs, constants or layer outputs
    /// based on a given name, the name may be suffixed with "_0", "_1" etc. if required
    /// </summary>
    internal string GetUniqueName(string name)
    {
        if (!ContainsName(name))
            return name;

        for (var i = 0;; i++)
        {
            var currentName = name + "_" + i;
            if (!ContainsName(currentName))
                return currentName;
        }
    }

    /// <summary>
    /// Checks if `name` is used in any model inputs, constants or layer outputs.
    /// </summary>
    bool ContainsName(string name)
    {
        if (constants.Any(constant => constant.name == name))
            return true;
        if (inputs.Any(input => input.name == name))
            return true;
        foreach (var layer in layers)
        {
            if (layer.name == name)
                return true;
            if (layer.outputs == null)
                continue;
            foreach (var output in layer.outputs)
            {
                if (output == name)
                    return true;
            }
        }
        if (outputs.Contains(name))
            return true;
        return false;
    }

    internal void ValidateInputTensorShape(Input input, TensorShape shape)
    {
        if (shape.rank != input.shape.rank)
        {
            D.LogWarning($"Given input shape: {shape} is not compatible with model input shape: {GetSymbolicTensorShapeAsString(input.shape)} for input: {input.name}");
            return;
        }

        for (var i = 0; i < shape.rank; i++)
        {
            if (input.shape[i] != shape[i])
                D.LogWarning($"Given input shape: {shape} has different dimension from model input shape: {GetSymbolicTensorShapeAsString(input.shape)} for input: {input.name} at axis: {i}");
        }
    }

    /// <summary>
    /// Adds an input to the model with a symbolic tensor shape.
    /// </summary>
    /// <param name="name">The name of the input.</param>
    /// <param name="dataType">The data type of the input.</param>
    /// <param name="shape">The `SymbolicTensorShape` of the input.</param>
    public void AddInput(string name, DataType dataType, SymbolicTensorShape shape)
    {
        inputs.Add(new Input { name = name, dataType = dataType, shape = shape });
    }

    /// <summary>
    /// Adds an input to the model with a tensor shape.
    /// </summary>
    /// <param name="name">The name of the input.</param>
    /// <param name="dataType">The data type of the input.</param>
    /// <param name="shape">The `TensorShape` of the input.</param>
    public void AddInput(string name, DataType dataType, TensorShape shape)
    {
        inputs.Add(new Input { name = name, dataType = dataType, shape = new SymbolicTensorShape(shape) });
    }

    /// <summary>
    /// Adds an output called `name` to the model.
    /// </summary>
    /// <param name="name">The name of the input.</param>
    public void AddOutput(string name)
    {
        if (!outputs.Contains(name))
            outputs.Add(name);
    }

    /// <summary>
    /// Appends a `layer` to the model.
    /// </summary>
    /// <param name="layer">The layer to append.</param>
    public void AddLayer(Layers.Layer layer)
    {
        layers.Add(layer);
    }

    /// <summary>
    /// Adds a `constant` to the model.
    /// </summary>
    /// <param name="constant">The constant to add.</param>
    public void AddConstant(Layers.Constant constant)
    {
        constants.Add(constant);
    }

    internal void SetRemapNamedDims(Dictionary<char, string> remapNamedDims)
    {
        RemapNamedDims = remapNamedDims;
    }

    internal void DisposeWeights()
    {
        foreach (var constant in constants)
            constant.weights?.Dispose();
    }
}
} // namespace Unity.Sentis
