using UnityEngine;
using UnityEditor.AssetImporters;
using System.IO;
using System.Runtime.CompilerServices;
using System.Collections.Generic;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]

namespace Unity.Sentis
{
    /// <summary>
    /// Represents an importer for serialized Sentis model files.
    /// </summary>
    [ScriptedImporter(1, new[] { "sentis" })]
    class SentisModelImporter : ScriptedImporter
    {
        /// <summary>
        /// Callback that Sentis calls when the ONNX model has finished importing.
        /// </summary>
        /// <param name="ctx">Asset import context</param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var model = ModelLoader.Load(ctx.assetPath);

            ModelAsset asset = ScriptableObject.CreateInstance<ModelAsset>();

            ModelAssetData modelAssetData = ScriptableObject.CreateInstance<ModelAssetData>();
            var descStream = new MemoryStream();
            ModelWriter.SaveModelDesc(descStream, model);
            modelAssetData.value = descStream.ToArray();
            modelAssetData.name = "Data";
            modelAssetData.hideFlags = HideFlags.HideInHierarchy;
            descStream.Close();
            descStream.Dispose();

            asset.modelAssetData = modelAssetData;

            var weightStreams = new List<MemoryStream>();
            ModelWriter.SaveModelWeights(weightStreams, model);

            asset.modelWeightsChunks = new ModelAssetWeightsData[weightStreams.Count];
            for (int i = 0; i < weightStreams.Count; i++)
            {
                var stream = weightStreams[i];
                asset.modelWeightsChunks[i] = ScriptableObject.CreateInstance<ModelAssetWeightsData>();
                asset.modelWeightsChunks[i].value = stream.ToArray();
                asset.modelWeightsChunks[i].name = "Data";
                asset.modelWeightsChunks[i].hideFlags = HideFlags.HideInHierarchy;

                ctx.AddObjectToAsset($"model data weights {i}", asset.modelWeightsChunks[i]);

                stream.Close();
                stream.Dispose();
            }

            ctx.AddObjectToAsset("main obj", asset);
            ctx.AddObjectToAsset("model data", modelAssetData);

            ctx.SetMainObject(asset);
            model.DisposeWeights();
        }
    }
}
