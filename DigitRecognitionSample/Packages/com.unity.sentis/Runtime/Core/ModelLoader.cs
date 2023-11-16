// #define DEBUG_TIMING
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;
using UnityEngine.Profiling;

[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]

namespace Unity.Sentis {

/// <summary>
/// Provides methods for loading models.
/// </summary>
public static class ModelLoader
{
    /// <summary>
    /// Converts a binary `ModelAsset` representation of a neural network to an object-oriented `Model` representation.
    /// </summary>
    /// <param name="modelAsset">The binary `ModelAsset` model</param>
    /// <returns>The loaded `Model`</returns>
    public static Model Load(ModelAsset modelAsset)
    {
        Model model = new Model();

        MemoryStream descStream = Open(modelAsset.modelAssetData.value);
        LoadModelDesc(descStream, ref model);
        descStream.Dispose();

        List<MemoryStream> weightStreams = new List<MemoryStream>();
        foreach(var weightsData in modelAsset.modelWeightsChunks)
        {
            weightStreams.Add(Open(weightsData.value));
        }

        LoadModelWeights(weightStreams, ref model);

        foreach (var stream in weightStreams)
            stream.Dispose();

        return model;
    }

    /// <summary>
    /// Loads a model that has been serialized to disk.
    /// </summary>
    /// <param name="path">The path of the binary serialized model</param>
    /// <returns>The loaded `Model`</returns>
    public static Model Load(string path)
    {
        using FileStream fileStream = File.Open(path, FileMode.Open);
        return Load(fileStream);
    }

    /// <summary>
    /// Loads a serialized model from a stream.
    /// </summary>
    /// <param name="stream">The `Stream` from which to load the binary serialized model</param>
    /// <returns>The loaded `Model`</returns>
    public static Model Load(Stream stream)
    {
        Model model = new Model();

        LoadModelDesc(stream, ref model);
        LoadModelWeights(stream, ref model);

        return model;
    }

    /// <summary>
    /// Loads a model description without the external weights from a serialized `ModelAsset`
    /// </summary>
    /// <param name="modelAsset">The serialized `ModelAsset` from which to load the model description</param>
    /// <param name="model">The model to set the loaded description of</param>
    public static void LoadModelDesc(ModelAsset modelAsset, ref Model model)
    {
        MemoryStream descStream = Open(modelAsset.modelAssetData.value);
        LoadModelDesc(descStream, ref model);
        descStream.Dispose();
    }

    /// <summary>
    /// Loads a model weights without the description from a serialized `ModelAsset`
    /// </summary>
    /// <param name="modelAsset">The serialized `ModelAsset` from which to load the model weights</param>
    /// <param name="model">The model to set the loaded description of</param>
    public static void LoadModelWeights(ModelAsset modelAsset, ref Model model)
    {
        List<MemoryStream> weightStreams = new List<MemoryStream>();
        foreach(var weightsData in modelAsset.modelWeightsChunks)
        {
            weightStreams.Add(Open(weightsData.value));
        }

        LoadModelWeights(weightStreams, ref model);

        foreach (var stream in weightStreams)
            stream.Dispose();
    }

    internal static void LoadModelDesc(Stream stream, ref Model model)
    {
        Profiler.BeginSample("Sentis.LoadModelDesc");

        if (model == null)
            model = new Model();

        int version = Read<int>(stream);
        if (version != Model.Version)
            throw new NotSupportedException($"Format version not supported: {version}, please reimport model.");

        int numberOfInputs = Read<int>(stream);
        for (var l = 0; l < numberOfInputs; ++l)
        {
            var input = ReadObject<Model.Input>(stream);
            model.inputs.Add(input);
        }

        model.outputs = ReadObject<List<string>>(stream);

        int numberOfLayers = Read<int>(stream);
        for (var l = 0; l < numberOfLayers; ++l)
        {
            Layers.Layer layer = ReadObject<Layers.Layer>(stream);
            model.layers.Add(layer);
        }

        int numberOfIConstants = Read<int>(stream);
        for (var l = 0; l < numberOfIConstants; ++l)
        {
            Layers.Constant constant = new Layers.Constant();
            constant.name           = ReadObject<string>(stream);
            constant.shape          = ReadObject<TensorShape>(stream);
            constant.dataType       = (DataType)Read<int>(stream);
            constant.offset         = Read<long>(stream);
            constant.length         = Read<int>(stream);
            model.constants.Add(constant);
        }

        // Importer Reporting
        model.IrSource = ReadObject<string>(stream);
        model.IrVersion = Read<long>(stream);
        model.ProducerName = ReadObject<string>(stream);
        model.DefaultOpsetVersion = Read<long>(stream);

        int numOpsetVersions = Read<int>(stream);
        for (var i = 0; i < numOpsetVersions; ++i)
        {
            model.OpsetDescriptions.Add(new Model.OpsetDescription
            {
                domain = ReadObject<string>(stream),
                version = Read<long>(stream),
            });
        }

        int numWarnings = Read<int>(stream);
        for (var i = 0; i < numWarnings; ++i)
        {
            model.Warnings.Add(new Model.ImporterWarning(ReadObject<string>(stream), (Model.WarningType)Read<int>(stream), ReadObject<string>(stream)));
        }

        int numMetadataProps = Read<int>(stream);
        for (var i = 0; i < numMetadataProps; ++i)
        {
            model.Metadata.Add(ReadObject<string>(stream), ReadObject<string>(stream));
        }

        int numMappedParams = Read<int>(stream);
        for (var i = 0; i < numMappedParams; ++i)
        {
            model.RemapNamedDims.Add(Read<char>(stream), ReadObject<string>(stream));
        }

        int numLayersCPUFallback = Read<int>(stream);
        for (var i = 0; i < numLayersCPUFallback; ++i)
        {
            model.LayerCPUFallback.Add(ReadObject<string>(stream));
        }

        Profiler.EndSample();
    }

    static void LoadModelWeights(List<MemoryStream> streams, ref Model model)
    {
        Profiler.BeginSample("Sentis.LoadModelWeights");

        int streamIndex = 0;
        MemoryStream stream = streams[streamIndex];

        var sizeOfDataItem = sizeof(float);

        int lstart = 0;
        int lcount = 0;
        long count = 0;
        long readLength = 0;

        // write constant data
        for (var l = 0; l < model.constants.Count; ++l)
        {
            var constant = model.constants[l];
            int memorySize = constant.length * sizeOfDataItem;
            if (readLength + (long)memorySize >= (long)Int32.MaxValue)
            {
                var sharedWeightsArray  = new NativeTensorArrayFromManagedArray(stream.GetBuffer(), 0, (int)count);
                for (var ll = lstart; ll < (lstart + lcount); ++ll)
                {
                    model.constants[ll].weights = sharedWeightsArray;
                }

                streamIndex++;
                stream = streams[streamIndex];
                readLength = 0;
                count = 0;
                lstart = (lstart + lcount);
                lcount = 0;
            }
            readLength += memorySize;
            count += constant.length;
            lcount += 1;
        }

        var sharedWeightsArray2  = new NativeTensorArrayFromManagedArray(stream.GetBuffer(), 0, (int)count);
        for (var ll = lstart; ll < (lstart + lcount); ++ll)
        {
            model.constants[ll].weights = sharedWeightsArray2;
        }

        Profiler.EndSample();
    }

    static void LoadModelWeights(Stream stream, ref Model model)
    {
        Profiler.BeginSample("Sentis.LoadModelWeights");

        var sizeOfDataItem = sizeof(float);

        int lstart = 0;
        int lcount = 0;
        long count = 0;
        long readLength = 0;

        // write constant data
        for (var l = 0; l < model.constants.Count; ++l)
        {
            var constant = model.constants[l];
            int memorySize = constant.length * sizeOfDataItem;
            if (readLength + (long)memorySize >= (long)Int32.MaxValue)
            {
                var byteArray = new byte[(int)count * sizeOfDataItem];
                stream.Read(byteArray, 0, (int)count * sizeOfDataItem);
                var sharedWeightsArray  = new NativeTensorArrayFromManagedArray(byteArray, 0, (int)count);
                for (var ll = lstart; ll < (lstart + lcount); ++ll)
                {
                    model.constants[ll].weights = sharedWeightsArray;
                }

                readLength = 0;
                count = 0;
                lstart = (lstart + lcount);
                lcount = 0;
            }
            readLength += memorySize;
            count += constant.length;
            lcount += 1;
        }

        var byteArray2 = new byte[(int)count * sizeOfDataItem];
        stream.Read(byteArray2, 0, (int)count * sizeOfDataItem);
        var sharedWeightsArray2  = new NativeTensorArrayFromManagedArray(byteArray2, 0, (int)count);
        for (var ll = lstart; ll < (lstart + lcount); ++ll)
        {
            model.constants[ll].weights = sharedWeightsArray2;
        }

        Profiler.EndSample();
    }

    static T Read<T>(Stream stream) where T : unmanaged
    {
        unsafe
        {
            Span<byte> arr = stackalloc byte[sizeof(T)];
            stream.Read(arr);
            T dst = default(T);
            fixed (byte* src = &arr[0])
            {
                Buffer.MemoryCopy(src, &dst, sizeof(T), sizeof(T));
            }
            return dst;
        }
    }

    static T ReadObject<T>(Stream stream)
    {
        var bf = new BinaryFormatter();
        var obj = bf.Deserialize(stream);
        return (T)obj;
    }

    static MemoryStream Open(byte[] bytes)
    {
        return new MemoryStream(bytes, 0, bytes.Length, false, true);
    }
}


} // namespace Unity.Sentis
