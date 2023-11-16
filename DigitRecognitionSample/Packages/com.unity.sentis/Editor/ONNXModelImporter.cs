using System;
using UnityEngine;
using UnityEditor;
using UnityEditor.AssetImporters;
using System.IO;
using System.Runtime.CompilerServices;
using Unity.Sentis.ONNX;
using System.Collections.Generic;
using System.Reflection;
using System.Diagnostics;

[assembly: InternalsVisibleTo("Unity.Sentis.EditorTests")]
[assembly: InternalsVisibleTo("Unity.Sentis.Tests")]

namespace Unity.Sentis
{
    /// <summary>
    /// Represents an importer for Open Neural Network Exchange (ONNX) files.
    /// </summary>
    [ScriptedImporter(56, new[] { "onnx" })]
    [HelpURL("https://docs.unity3d.com/Packages/com.unity.sentis@latest/index.html")]
    class ONNXModelImporter : ScriptedImporter
    {
        // Configuration
        /// <summary>
        /// Whether Sentis optimizes the ONNX model during import. You can also change this setting in the Model Asset Import Settings in the Editor.
        /// </summary>
        public bool optimizeModel = true;

        /// <summary>
        /// Callback that Sentis calls when the ONNX model has finished importing.
        /// </summary>
        /// <param name="ctx">Asset import context</param>
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var converter = new ONNXModelConverter(optimizeModel, ctx.assetPath);
            converter.CollectOpImporters += CollectOpImporters;
            var model = converter.Convert();

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

        private void CollectOpImporters(Dictionary<string, IOpImporter> customOps)
        {
            if (customOps == null)
                customOps = new Dictionary<string, IOpImporter>();
            var types = TypeCache.GetTypesDerivedFrom<IOpImporter>();
            foreach (var type in types)
            {
                OpImportAttribute attribute = (OpImportAttribute)Attribute.GetCustomAttribute(type, typeof(OpImportAttribute));
                if (attribute == null)
                    continue;
                ConstructorInfo constructor = type.GetConstructor(Type.EmptyTypes);
                object opImporterObject = constructor.Invoke(new object[] { });
                customOps.Add(attribute.opType, opImporterObject as IOpImporter);
            }
        }
    }
}
