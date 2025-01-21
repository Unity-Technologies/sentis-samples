using System.Collections;
using UnityEngine;
using Unity.Sentis;

public class InferenceWebcam : MonoBehaviour
{
    public ModelAsset estimationModel;
    Worker m_engineEstimation;
    WebCamTexture webcamTexture;
    Tensor<float> inputTensor;
    RenderTexture outputTexture;

    public Material material;
    public Texture2D colorMap;

    int modelLayerCount = 0;
    public int framesToExectute = 2;

    void Start()
    {
        Application.targetFrameRate = 60;
        var model = ModelLoader.Load(estimationModel);

        // Post process
        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model);
        var outputs = Functional.Forward(model, inputs);
        var output = outputs[0];

        var max0 = Functional.ReduceMax(output, new[] { 1, 2 }, false);
        var min0 = Functional.ReduceMin(output, new[] { 1, 2 }, false);
        output = (output - min0) / (max0 - min0);

        model = graph.Compile(output);
        modelLayerCount = model.layers.Count;

        m_engineEstimation = new Worker(model, BackendType.GPUCompute);

        WebCamDevice[] devices = WebCamTexture.devices;
        webcamTexture = new WebCamTexture(1920, 1080);
        webcamTexture.deviceName = devices[0].name;
        webcamTexture.Play();

        outputTexture = new RenderTexture(256, 256, 0, RenderTextureFormat.ARGBFloat);
        inputTensor = new Tensor<float>(new TensorShape(1, 3, 256, 256));
    }

    bool executionStarted = false;
    IEnumerator executionSchedule;
    void Update()
    {
        if (!executionStarted)
        {
            TextureConverter.ToTensor(webcamTexture, inputTensor, new TextureTransform());
            executionSchedule = m_engineEstimation.ScheduleIterable(inputTensor);
            executionStarted = true;
        }

        bool hasMoreWork = false;
        int layersToRun = (modelLayerCount + framesToExectute - 1) / framesToExectute; // round up
        for (int i = 0; i < layersToRun; i++)
        {
            hasMoreWork = executionSchedule.MoveNext();
            if (!hasMoreWork)
                break;
        }

        if (hasMoreWork)
            return;

        var output = m_engineEstimation.PeekOutput() as Tensor<float>;
        output.Reshape(output.shape.Unsqueeze(0));
        TextureConverter.RenderToTexture(output, outputTexture, new TextureTransform().SetCoordOrigin(CoordOrigin.BottomLeft));
        executionStarted = false;
    }

    void OnRenderObject()
    {
        material.SetVector("ScreenCamResolution", new Vector4(Screen.height, Screen.width, 0, 0));
        material.SetTexture("WebCamTex", webcamTexture);
        material.SetTexture("DepthTex", outputTexture);
        material.SetTexture("ColorRampTex", colorMap);
        Graphics.Blit(null, material);
    }

    void OnDestroy()
    {
        m_engineEstimation.Dispose();
        inputTensor.Dispose();
        outputTexture.Release();
    }
}
