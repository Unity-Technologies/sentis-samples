using System.IO;
using Unity.InferenceEngine.Samples.TTS.Assets;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace Unity.InferenceEngine.Samples.TTS.Editor
{
    [ScriptedImporter(1, new[] { "bin" })]
    public class BinFileImporter : ScriptedImporter
    {
        public override void OnImportAsset(AssetImportContext ctx)
        {
            var bytes = File.ReadAllBytes(ctx.assetPath);
            var bytesAsset = ScriptableObject.CreateInstance<RawBytesAsset>();
            bytesAsset.bytes = bytes;

            ctx.AddObjectToAsset("main", bytesAsset);
            ctx.SetMainObject(bytesAsset);
        }
    }
}
