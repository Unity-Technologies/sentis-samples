using System;

/// <summary>
/// DataStructures to implement OpenInference with UnityWebRequest
/// https://github.com/kserve/open-inference-protocol/blob/main/specification/protocol/inference_rest.md
/// </summary>

[Serializable]
struct NetworkTensorShape
{
    public string name;
    public string datatype;
    public int[] shape;
};

[Serializable]
struct EmptyRequest
{

};

[Serializable]
struct ModelMetaDataResponse
{
    public string name;
    public string[] versions;
    public NetworkTensorShape[] inputs;
    public NetworkTensorShape[] outputs;
    public string platform;
};
[Serializable]
struct Parameters
{
    public int binary_data_size;
}
[Serializable]
struct NetworkInput
{
    public string name;
    public int[] shape;
    public string datatype;
    public Parameters parameters;
};
[Serializable]
struct OutputRequestParameters
{
    public bool binary_data;
}
[Serializable]
struct OutputRequest
{
    public string name;
    public OutputRequestParameters parameters;
};

[Serializable]
struct InferenceRequest
{
    public NetworkInput[] inputs;
    public OutputRequest[] outputs;
};

[Serializable]
struct NetworkOutput
{
    public string name;
    public int[] shape;
    public string datatype;
    public Parameters parameters;
    // The two below fields are not used (not filled by serialization) when binary output is picked
    // Only one field is valid and we rename the json data to route into the right field
    public float[] data;
    public int[] zzin;
};

[Serializable]
struct InferenceResponse
{
    public string model_name;
    public string model_version;
    public string id;
    public NetworkOutput[] outputs;
};

// UGS-specific section

[Serializable]
struct ServerInstance
{
    public string host;
    public int locationId;
    public string locationName;
}

[Serializable]
struct DiscoveryResponse
{
    public string name;
    public ServerInstance[] servers;
    public string version;
}
