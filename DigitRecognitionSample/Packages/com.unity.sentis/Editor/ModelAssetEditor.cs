using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;
using System;
using UnityEngine.UIElements;
using Unity.Sentis.Compiler.Analyser;
using System.IO;

namespace Unity.Sentis.Editor
{
[CustomEditor(typeof(ModelAsset))]
public class ModelAssetEditor : UnityEditor.Editor
{
    Model m_Model;

    Foldout CreateFoldoutListView(List<string> items, string name)
    {
        Func<VisualElement> makeItem = () => new Label();
        Action<VisualElement, int> bindItem = (e, i) => (e as Label).text = items[i];

        var listView = new ListView(items, 16, makeItem, bindItem);
        listView.showAlternatingRowBackgrounds = AlternatingRowBackground.All;
        listView.showBorder = true;
        listView.selectionType = SelectionType.Multiple;
        listView.style.flexGrow = 1;

        var inputMenu = new Foldout();
        inputMenu.text = name;
        inputMenu.style.maxHeight = 400;
        inputMenu.Add(listView);

        return inputMenu;
    }

    ListView CreateListView(List<string> items)
    {
        Func<VisualElement> makeItem = () => new Label();
        Action<VisualElement, int> bindItem = (e, i) => (e as Label).text = items[i];

        var listView = new ListView(items, 16, makeItem, bindItem);
        listView.showAlternatingRowBackgrounds = AlternatingRowBackground.All;
        listView.showBorder = true;
        listView.selectionType = SelectionType.Multiple;
        listView.style.flexGrow = 1;

        return listView;
    }

    void CreateWarningsListView(VisualElement rootElement)
    {
        if (!m_Model.Warnings.Any())
            return;

        var warningsError = new List<string>();
        var warningsWarning = new List<string>();
        var warningsInfo = new List<string>();
        var warningsNeutral = new List<string>();
        foreach (var warning in m_Model.Warnings)
        {
            var warningDesc = warning.Message;
            Model.WarningType messageType = warning.MessageSeverity;

            switch (messageType)
            {
                case Model.WarningType.None:
                    warningsNeutral.Add($"<b>{warning}</b>: {warningDesc}");
                    break;
                case Model.WarningType.Info:
                    warningsInfo.Add($"<b>{warning}</b>: {warningDesc}");
                    break;
                case Model.WarningType.Warning:
                    warningsWarning.Add($"<b>{warning}</b>: {warningDesc}");
                    break;
                case Model.WarningType.Error:
                    warningsError.Add($"<b>{warning}</b>: {warningDesc}");
                    break;
            }
        }

        bool warnings = warningsError.Any() || warningsWarning.Any() || warningsInfo.Any() || warningsNeutral.Any();

        if (warningsError.Any())
        {
            rootElement.Add(new HelpBox("Model contains errors. Model will not run.\n", HelpBoxMessageType.Error));
            rootElement.Add(CreateListView(warningsError));
        }
        if (warningsWarning.Any())
        {
            rootElement.Add(new HelpBox("Model contains warnings. Behavior might be incorrect.\n", HelpBoxMessageType.Warning));
            rootElement.Add(CreateListView(warningsWarning));
        }
        if (warningsInfo.Any())
        {
            rootElement.Add(new HelpBox("Model contains import information.\n", HelpBoxMessageType.Info));
            rootElement.Add(CreateListView(warningsInfo));
        }
        if (warningsNeutral.Any())
        {
            rootElement.Add(new HelpBox("Comments: \n", HelpBoxMessageType.None));
            rootElement.Add(CreateListView(warningsNeutral));
        }
        if (warnings)
        {
            var box = new Box();
            box.style.height = 20;
            var color = box.style.backgroundColor.value;
            box.style.backgroundColor = new Color(color.r, color.g, color.b, 0.0f);
            rootElement.Add(box);
        }
    }

    void CreateMetadataListView(VisualElement rootElement)
    {
        if (!m_Model.Metadata.Any())
            return;

        var metadata = m_Model.Metadata;
        var items = new List<string>(metadata.Count);
        foreach (var keyval in m_Model.Metadata)
            items.Add($"<b>{keyval.Key}</b> {keyval.Value}");

        var inputMenu = CreateFoldoutListView(items, $"<b>Metadata</b>");
        rootElement.Add(inputMenu);
    }

    void CreateInputListView(VisualElement rootElement)
    {
        var inputs = m_Model.inputs;
        var items = new List<string>(inputs.Count);
        foreach (var input in inputs)
            items.Add($"<b>{input.name}</b> {m_Model.GetSymbolicTensorShapeAsString(input.shape)}, {input.dataType}");

        var inputMenu = CreateFoldoutListView(items, $"<b>Inputs ({inputs.Count})</b>");
        rootElement.Add(inputMenu);
    }

    void CreateOutputListView(VisualElement rootElement)
    {
        var outputs = m_Model.outputs;
        var items = new List<string>(outputs.Count);
        try
        {
            var ctx = PartialInferenceAnalysis.InferModelPartialTensors(m_Model, false);
            foreach (var output in outputs)
            {
                var partialTensor = ctx.GetPartialTensor(output);
                items.Add($"<b>{output}</b> {m_Model.GetSymbolicTensorShapeAsString(partialTensor.shape)}, {partialTensor.dataType}");
            }
        }
        catch (Exception)
        {
            foreach (var output in outputs)
                items.Add($"<b>{output}</b>");
        }
        var inputMenu = CreateFoldoutListView(items, $"<b>Outputs ({outputs.Count})</b>");
        rootElement.Add(inputMenu);
    }

    void CreateLayersListView(VisualElement rootElement)
    {
        var layers = m_Model.layers;
        var items = new List<string>(layers.Count);
        foreach (var layer in layers)
        {
            string ls = layer.ToString();
            items.Add($"<b>{layer.profilerTag}</b> {ls.Substring(ls.IndexOf('-') + 2)}");
        }

        var layerMenu = CreateFoldoutListView(items, $"<b>Layers ({layers.Count})</b>");
        rootElement.Add(layerMenu);
    }

    void CreateConstantsListView(VisualElement rootElement)
    {
        long totalWeightsSizeInBytes = 0;
        var constants = m_Model.constants;
        var items = new List<string>(constants.Count);
        foreach (var constant in constants)
        {
            string cs = constant.ToString();
            items.Add($"<b>Constant</b> {cs.Substring(cs.IndexOf('-') + 2)}");
            totalWeightsSizeInBytes += constant.length * sizeof(float);
        }

        var constantsMenu = CreateFoldoutListView(items, $"<b>Constants ({constants.Count})</b>");
        rootElement.Add(constantsMenu);
        rootElement.Add(new Label($"Total weight size: {totalWeightsSizeInBytes / (1024 * 1024):n0} MB"));
    }

    public void LoadAndSerializeModel(ModelAsset modelAsset, string name)
    {
        Model model = new Model();
        ModelLoader.LoadModelDesc(modelAsset, ref model);
        ModelLoader.LoadModelWeights(modelAsset, ref model);
        if (!Directory.Exists(Application.streamingAssetsPath))
            Directory.CreateDirectory(Application.streamingAssetsPath);
        string fullpath = Path.Combine(Application.streamingAssetsPath, $"{name}.sentis");
        ModelWriter.Save(fullpath, model);
        AssetDatabase.Refresh();
        model.DisposeWeights();
    }

    void CreateSerializeButton(VisualElement rootElement, ModelAsset modelAsset, string name)
    {
        var button = new Button(() => LoadAndSerializeModel(modelAsset, name));
        button.text = "Serialize To StreamingAssets";
        rootElement.Add(button);
    }

    public override VisualElement CreateInspectorGUI()
    {
        var rootInspector = new VisualElement();

        var modelAsset = target as ModelAsset;
        if (modelAsset == null)
            return rootInspector;
        if (modelAsset.modelAssetData == null)
            return rootInspector;

        ModelLoader.LoadModelDesc(modelAsset, ref m_Model);

        CreateWarningsListView(rootInspector);
        CreateSerializeButton(rootInspector, modelAsset, target.name);
        CreateMetadataListView(rootInspector);
        CreateInputListView(rootInspector);
        CreateOutputListView(rootInspector);
        CreateLayersListView(rootInspector);
        CreateConstantsListView(rootInspector);

        rootInspector.Add(new Label($"Source: {m_Model.IrSource}"));
        rootInspector.Add(new Label($"Default Opset Version: {m_Model.DefaultOpsetVersion}"));
        rootInspector.Add(new Label($"Producer Name: {m_Model.ProducerName}"));

        return rootInspector;
    }
}
}
