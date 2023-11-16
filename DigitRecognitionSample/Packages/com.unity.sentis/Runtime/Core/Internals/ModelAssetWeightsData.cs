using UnityEngine;

namespace Unity.Sentis
{
    /// <summary>
    /// Represents data storage for the constant weights of a model.
    /// </summary>
    [PreferBinarySerialization]
    public class ModelAssetWeightsData : ScriptableObject
    {
        /// <summary>
        /// The serialized byte array of the data.
        /// </summary>
        [HideInInspector]
        public byte[] value;
    }
}
