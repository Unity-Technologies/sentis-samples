using System;
using System.Collections.Generic;
using System.Text;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Sentis.Layers;
using UnityEngine;
using UnityEngine.Networking;

namespace Unity.Sentis
{
[Serializable]
struct InputDescription
{
    public string name;
    public DataType type;
    public SymbolicTensorShape shape;
}

class CloudUtil
{
    internal static Dictionary<string, DataType> typeFromName = new Dictionary<string, DataType>{
        {"INT32", DataType.Int},
        {"FP32", DataType.Float},
    };

    public static InferenceRequest GetRequestObject(ModelMetaDataResponse metaDataResponse)
    {
        var requestObject = new InferenceRequest();
        requestObject.inputs = new NetworkInput[metaDataResponse.inputs.Length];

        for (int i = 0; i < metaDataResponse.inputs.Length; i++)
        {
            requestObject.inputs[i].name = metaDataResponse.inputs[i].name;
            requestObject.inputs[i].shape = metaDataResponse.inputs[i].shape;
            requestObject.inputs[i].datatype = metaDataResponse.inputs[i].datatype;
            requestObject.inputs[i].parameters = new Parameters()
                {binary_data_size = new TensorShape(metaDataResponse.inputs[i].shape).length * sizeof(float)};
        }

        requestObject.outputs = new OutputRequest[metaDataResponse.outputs.Length];

        for (int i = 0; i < metaDataResponse.outputs.Length; i++)
        {
            requestObject.outputs[i].name = metaDataResponse.outputs[i].name;
            requestObject.outputs[i].parameters = new OutputRequestParameters() {binary_data = true};
        }

        return requestObject;
    }
};

[Serializable]
class Cloud : Layer
{
    internal const string kContentLengthHeader =  "Inference-Header-Content-Length";
    private const string kContentTypeHeader = "Content-Type";
    private const string kJsonType = "application/json";

    public string modelName;

    public InputDescription[] inputList;

    private int[] inputByteSizes;
    private int[] outputByteSizes;

    [System.NonSerialized]
    public ICloudAccessProvider accessProvider;

    private Type CloudAccessProviderType;

    private TensorShape[] outputShapes;
    private DataType[] outputTypes;
    private InferenceRequest requestObject;
    private int inputDataSize = 0;

    public Cloud(ICloudAccessProvider accessProvider, string modelName)
    {
        this.accessProvider = accessProvider;
        this.modelName = modelName;
    }

    internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
    {
        var ret = new PartialTensor(outputTypes[0], new SymbolicTensorShape(outputShapes[0]));

        // first one is added above, so let's start at 1
        for (int index = 1; index < outputs.Length; index++)
        {
            ctx.AddPartialTensor(outputs[index], new PartialTensor(outputTypes[index], new SymbolicTensorShape(outputShapes[index])));
        }

        return ret;
    }

    // todo: this is probably the right place to check input names
    private void PrepareInputOutput(ModelMetaDataResponse metaDataResponse)
    {
        int outputDataSize = 0;
        inputDataSize = 0;

        inputByteSizes = new int[inputs.Length];
        outputByteSizes = new int[outputs.Length];
        outputShapes = new TensorShape[metaDataResponse.outputs.Length];
        outputTypes = new DataType[metaDataResponse.outputs.Length];

        // yeah this is 4, but let's be explicit that we store either
        int fieldSize = Math.Max(sizeof(float), sizeof(int));

        for (int index = 0; index < inputs.Length; index++)
        {
            inputByteSizes[index] = new TensorShape(metaDataResponse.inputs[index].shape).length * fieldSize;
            inputDataSize += inputByteSizes[index];
        }

        for (int index = 0; index < metaDataResponse.outputs.Length; index++)
        {
            outputShapes[index] = new TensorShape(metaDataResponse.outputs[index].shape);
            outputByteSizes[index] = outputShapes[index].length * fieldSize;
            outputDataSize += outputByteSizes[index];
            outputTypes[index] = CloudUtil.typeFromName[metaDataResponse.outputs[index].datatype];
        }

        requestObject = CloudUtil.GetRequestObject(metaDataResponse);
    }

    public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
    {
        int currPos = 0;

        UnityWebRequest request = new UnityWebRequest(accessProvider.InferenceUrl(modelName), UnityWebRequest.kHttpVerbPOST);
        request.SetRequestHeader(kContentTypeHeader, kJsonType);

        var requestString = JsonUtility.ToJson(requestObject);
        byte[] myData = Encoding.UTF8.GetBytes(requestString);

        request.SetRequestHeader(kContentLengthHeader, myData.Length.ToString());


        int startPosition = myData.Length;
        Array.Resize(ref myData, myData.Length + inputDataSize);

        currPos = startPosition;

        for (var inputIndex = 0; inputIndex < inputs.Length; inputIndex++)
        {
            inputTensors[inputIndex].MakeReadable();

            unsafe
            {
                var src = ArrayTensorData.Pin(inputTensors[inputIndex]).array.RawPtr;
                fixed (byte* dst = myData)
                {
                    UnsafeUtility.MemCpy(dst + currPos, src, inputByteSizes[inputIndex]);
                }
            }
            currPos += inputByteSizes[inputIndex];
        }

        request.uploadHandler = new UploadHandlerRaw(myData);
        request.downloadHandler = new DownloadHandlerBuffer();

        AsyncWebReadbackRequest requestData = new AsyncWebReadbackRequest();
        requestData.asyncOperation = request.SendWebRequest();

        Tensor ret = null;
        var offset = 0;

        for (var outputIndex = 0; outputIndex < outputs.Length; outputIndex++)
        {
            CloudTensorData cloudTensorData = new CloudTensorData(outputShapes[0].length, offset, requestData);
            Tensor t = ctx.backend.NewTensor(outputShapes[outputIndex], outputTypes[outputIndex], AllocScope.LayerOutput);
            t.AttachToDevice(cloudTensorData);

            if (outputIndex == 0) ret = t;
            else ctx.vars.Store(outputs[outputIndex], t);

            offset += outputByteSizes[outputIndex];
        }

        return ret;
    }

    public void GetMetadata()
    {
        UnityWebRequest metadataRequest;

        metadataRequest = new UnityWebRequest(accessProvider.MetadataUrl(modelName), UnityWebRequest.kHttpVerbGET);
        metadataRequest.SetRequestHeader(kContentTypeHeader, kJsonType);

        foreach (var p in accessProvider.Headers())
        {
            metadataRequest.SetRequestHeader(p.Key, p.Value);
        }

        var requestObject = new EmptyRequest();
        var requestString = JsonUtility.ToJson(requestObject);
        byte[] myData = Encoding.UTF8.GetBytes(requestString);

        using (metadataRequest.uploadHandler = new UploadHandlerRaw(myData))
        using (metadataRequest.downloadHandler = new DownloadHandlerBuffer())
        {
            var asyncOperation = metadataRequest.SendWebRequest();

            while (!asyncOperation.isDone)
            {
            }

            string responseText = metadataRequest.downloadHandler.text;

            var jsonResponse = JsonUtility.FromJson<ModelMetaDataResponse>(responseText);

            bool success = (metadataRequest.result == UnityWebRequest.Result.Success);

            if (success)
            {
                inputList = new InputDescription[jsonResponse.inputs.Length];

                string[] inputNames = new string[jsonResponse.inputs.Length];
                string[] outputNames = new string[jsonResponse.outputs.Length];

                for (int i = 0; i < jsonResponse.inputs.Length; i++)
                {
                    inputList[i] = new InputDescription
                    {
                        name = jsonResponse.inputs[i].name, type = CloudUtil.typeFromName[jsonResponse.inputs[i].datatype],
                        shape = new SymbolicTensorShape(new TensorShape(jsonResponse.inputs[i].shape))
                    };
                    inputNames[i] = jsonResponse.inputs[i].name;
                }

                for (int i = 0; i < jsonResponse.outputs.Length; i++)
                {
                    outputNames[i] = jsonResponse.outputs[i].name;
                }

                name = outputNames[0];
                inputs = new string[inputNames.Length];
                inputNames.CopyTo(inputs, 0);
                outputs = new string[outputNames.Length];
                outputNames.CopyTo(outputs, 0);

                CloudAccessProviderType = accessProvider.GetType();
                PrepareInputOutput(jsonResponse);
            }
        }
    }

    public void RestoreProvider()
    {
        accessProvider = (ICloudAccessProvider)Activator.CreateInstance(CloudAccessProviderType);
    }

    internal override string profilerTag => "Cloud";
}
}
