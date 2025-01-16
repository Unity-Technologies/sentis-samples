using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using System.Linq;

/*
 *  Neural net engine and handles the inference.
 *   - Shifts the image to the center for better inference.
 *   (The model was trained on images centered in the texture this will give better results)
 *  - recentering of the image is also done using special operations on the GPU
 *
 */

// current bounds of the drawn image. It will help if we need to recenter the image later
public struct Bounds
{
    public int left;
    public int right;
    public int top;
    public int bottom;
}

public class MNISTEngine : MonoBehaviour
{
    public ModelAsset mnistONNX;

    // engine type
    Worker engine;

    // This small model works just as fast on the CPU as well as the GPU:
    static Unity.Sentis.BackendType backendType = Unity.Sentis.BackendType.GPUCompute;

    // width and height of the image:
    const int imageWidth = 28;

    // input tensor
    Tensor<float> inputTensor = null;

    Camera lookCamera;


    void Start()
    {
        // load the neural network model from the asset:
        Model model = ModelLoader.Load(mnistONNX);

        var graph = new FunctionalGraph();
        inputTensor = new Tensor<float>(new TensorShape(1, 1, imageWidth, imageWidth));
        var input = graph.AddInput(DataType.Float, new TensorShape(1, 1, imageWidth, imageWidth));
        var outputs = Functional.Forward(model, input);
        var result = outputs[0];

        // Convert the result to probabilities between 0..1 using the softmax function
        var probabilities = Functional.Softmax(result);
        var indexOfMaxProba = Functional.ArgMax(probabilities, -1, false);
        model = graph.Compile(probabilities, indexOfMaxProba);

        // create the neural network engine:
        engine = new Worker(model, backendType);

        //The camera which we'll be using to calculate the rays on the image:
        lookCamera = Camera.main;
    }

    // Sends the image to the neural network model and returns the probability that the image is each particular digit.
    public (float, int) GetMostLikelyDigitProbability(Texture2D drawableTexture)
    {
        inputTensor?.Dispose();

        // Convert the texture into a tensor, it has width=W, height=W, and channels=1:
        inputTensor = TextureConverter.ToTensor(drawableTexture, imageWidth, imageWidth, 1);

        // run the neural network:
        engine.Schedule(inputTensor);

        // We get a reference to the outputs of the neural network. Make the result from the GPU readable on the CPU
        using var probabilities = (engine.PeekOutput(0) as Tensor<float>).ReadbackAndClone();
        using var indexOfMaxProba = (engine.PeekOutput(1) as Tensor<int>).ReadbackAndClone();

        var predictedNumber = indexOfMaxProba[0];
        var probability = probabilities[predictedNumber];

        return (probability, predictedNumber);
    }

    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            MouseClicked();
        }
        else if (Input.GetMouseButton(0))
        {
            MouseIsDown();
        }
    }

    // Detect the mouse click and send the info to the panel class
    void MouseClicked()
    {
        Ray ray = lookCamera.ScreenPointToRay(Input.mousePosition);
        if (Physics.Raycast(ray, out RaycastHit hit) && hit.collider.name == "Screen")
        {
            Panel panel = hit.collider.GetComponentInParent<Panel>();
            if (!panel) return;
            panel.ScreenMouseDown(hit);
        }
    }

    // Detect if the mouse is down and sent the info to the panel class
    void MouseIsDown()
    {
        Ray ray = lookCamera.ScreenPointToRay(Input.mousePosition);
        if (Physics.Raycast(ray, out RaycastHit hit) && hit.collider.name == "Screen")
        {
            Panel panel = hit.collider.GetComponentInParent<Panel>();
            if (!panel) return;
            panel.ScreenGetMouse(hit);
        }
    }

    // Clean up all our resources at the end of the session so we don't leave anything on the GPU or in memory:
    private void OnDestroy()
    {
        inputTensor?.Dispose();
        engine?.Dispose();
    }

}
