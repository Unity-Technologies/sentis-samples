using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;

public class InferenceWebcam : MonoBehaviour
{
    public ModelAsset estimationModel;
    IWorker m_engineEstimation;
    WebCamTexture webcamTexture;
    TensorFloat inputTensor;
    RenderTexture outputTexture;

    public Material material;
    public Texture2D colorMap;

    int modelLayerCount = 0;
    public int framesToExectute = 2;
    
    void Start()
    {
        Application.targetFrameRate = 60;
        var model = ModelLoader.Load(estimationModel);
        var output = model.outputs[0];
        model.layers.Add(new Unity.Sentis.Layers.ReduceMax("max0", new[] { output }, false));
        model.layers.Add(new Unity.Sentis.Layers.ReduceMin("min0", new[] { output }, false));
        model.layers.Add(new Unity.Sentis.Layers.Sub("maxO - minO", "max0", "min0"));
        model.layers.Add(new Unity.Sentis.Layers.Sub("output - min0", output, "min0"));
        model.layers.Add(new Unity.Sentis.Layers.Div("output2", "output - min0", "maxO - minO"));
        modelLayerCount = model.layers.Count;
        model.outputs = new List<string>() { "output2" };
        m_engineEstimation = WorkerFactory.CreateWorker(BackendType.GPUCompute, model);

        WebCamDevice[] devices = WebCamTexture.devices;
        webcamTexture = new WebCamTexture(1920, 1080);
        webcamTexture.deviceName = devices[0].name;
        webcamTexture.Play();

        outputTexture = new RenderTexture(256, 256, 0, RenderTextureFormat.ARGBFloat);
        inputTensor = TensorFloat.Zeros(new TensorShape(1, 3, 256, 256));
    }

    bool executionStarted = false;
    IEnumerator executionSchedule;
    private void Update()
    {
        if (!executionStarted)
        {
            TextureConverter.ToTensor(webcamTexture, inputTensor, new TextureTransform());
            executionSchedule = m_engineEstimation.StartManualSchedule(inputTensor);
            executionStarted = true;
        }

        bool hasMoreWork = false;
        for (int i = 0; i < modelLayerCount / framesToExectute; i++)
        {
            hasMoreWork = executionSchedule.MoveNext();
            if (!hasMoreWork)
                break;
        }

        if (hasMoreWork)
            return;

        var output = m_engineEstimation.PeekOutput() as TensorFloat;
        output = output.ShallowReshape(output.shape.Unsqueeze(0)) as TensorFloat;
        TextureConverter.RenderToTexture(output as TensorFloat, outputTexture, new TextureTransform().SetCoordOrigin(CoordOrigin.BottomLeft));
        executionStarted = false;
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        material.SetVector("ScreenCamResolution", new Vector4(Screen.height, Screen.width, 0, 0));
        material.SetTexture("WebCamTex", webcamTexture);
        material.SetTexture("DepthTex", outputTexture);
        material.SetTexture("ColorRampTex", colorMap);
        Graphics.Blit(null, destination, material);
    }

    private void OnDestroy()
    {
        m_engineEstimation.Dispose();
        inputTensor.Dispose();
        outputTexture.Release();
    }
}
