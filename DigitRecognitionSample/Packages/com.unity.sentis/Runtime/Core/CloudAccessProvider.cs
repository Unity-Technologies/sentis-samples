using System;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace Unity.Sentis {

interface ICloudAccessProvider
{
    // The implementer of a specific provider must implement the following methods:

    // Return the http headers needed to connect to their service
    // For example, one of the required headers might be:
    //     key: "Authorization", value: "Hunter2"
    // where the value is the secret password provide by this accessor
    Dictionary<string, string> Headers();

    // Returns the URL to use for inference.
    // The returned URL will be used as-is, when a call is made to perform inference
    // "v2" is provided as a default version since OpenInference specifies
    // v2/models/<model_name>/infer as inference URL
    string InferenceUrl(string modelName, string modelVersion = "v2");

    // Returns the URL to use for metadata.
    // The returned URL will be used as-is, when a call is made to query metadata
    // "v2" is provided as a default version since OpenInference specifies
    // v2/models/<model_name> as metadata URL
    string MetadataUrl(string modelName, string modelVersion = "v2");

    // Return an array of servers
    // The implementor of the accessor must return the list of servers that the client can connect to
    ServerInstance[] Servers();

    // Allows user code to pick the server it wants to use in the list returned by Servers();
    void SelectServer(int index);
}
} // namespace Unity.Sentis
