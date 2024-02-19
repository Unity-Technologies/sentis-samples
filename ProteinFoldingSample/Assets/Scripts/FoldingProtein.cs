using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;

/*
 *  This uses gradient descent to fold the protein molecule up to match the distance matrix as output from the Alphafold model.
 *  
 *  This is the secondary part of the algorithm. The main Alphafold model neural network gives the distance matrix.
 *  Then we have to use this matrix to estimate how the protein folds up.
 */
public class FoldingProtein : MonoBehaviour
{

    public ModelAsset onnx;
    IWorker worker;
    static Ops ops;
    ITensorAllocator m_Allocator;

    public RibbonDrawer ribbon; //This is the class which draws the ribbon

    public static int N = 64; //number of amnio acids in sequence

    TensorFloat distogram, distogramSquared;
    TensorFloat positions; //positions of amino acids
    const float startingS = 0.015f;
    float s = startingS;
    
    public GameObject protein;

    public RawImage outputImage, distancesImage;

    public Text energy;

    public Texture2D blankTexture;

    //public static FoldingProtein instance;

    public static bool go = false; //This controls if the ribbon is being updated

    // Start is called before the first frame update
    void Start()
    {
      //  instance = this;

        m_Allocator = new TensorCachingAllocator();
        ops = WorkerFactory.CreateOps(BackendType.GPUCompute, m_Allocator);

        Model model = ModelLoader.Load(onnx);
        worker = model.CreateWorker(Unity.Sentis.DeviceType.GPU);
    }

    //The folding is done by a gradient descent model which moves the positions, x, by dx, to get to an energy minimum
    //The model can be folded up multiple times to try and get a folding with a low energy.
    void DoFolding()
    {
        DrawCurrentDistances();

        var inputs = new Dictionary<string, Tensor>
        {
            {"x", positions },
            {"D", distogramSquared },
        };

        worker.Execute(inputs);

        //Display the current energy
        using TensorFloat energyTensor = worker.PeekOutput("energy") as TensorFloat;
        energyTensor.MakeReadable();
        float e = energyTensor[0] / N;// 100f;
        energy.text = "Reconstructing protein\nLoss: " + e;

        using TensorFloat dx = worker.PeekOutput("dx") as TensorFloat;

        //psudo code: xvals = clamp(xvals+s*clamp(dxvals,-0.1/s,0.1/s),-100,100)
        using TensorFloat dxclip = ops.Clip(dx, -0.1f / s, 0.1f / s);
        using TensorFloat dxclip_s = ops.Mul(dxclip, s);
        using TensorFloat newX = ops.Sum(new[] { positions, dxclip_s });
        TensorHelper.Replace(ref positions, ops.Clip(newX, -100f, 100f));
    }

    public void DrawDistogram(TensorFloat distogram)
    {
        using TensorFloat d2 = distogram.DeepCopy().ShallowReshape(new TensorShape(1, 1, N, N)) as TensorFloat;
        Texture tex = TextureConverter.ToTexture(d2);
        tex.filterMode = FilterMode.Point;
        (outputImage.texture as RenderTexture)?.Release();
        outputImage.texture = tex;
    }

    public void UpdateDistogram(TensorFloat d)
    {
        TensorHelper.Replace(ref distogram, d);
    }

    public void UpdateDistogramSquared(TensorFloat d)
    {
        TensorHelper.Replace(ref distogramSquared, d);
    }

    public void Stop()
    {
        go = false;
        energy.text = "";
        distancesImage.texture = blankTexture;
        ribbon.gameObject.SetActive(false);
    }


    void Randomize()
    {
        TensorHelper.Replace(ref positions, ops.RandomUniform(new TensorShape(1, N, 3), -3f, 3f, (float)Time.time));
        s = startingS;
    }

    //Randomly shift amino acid positions. Can sometimes help to get out of local minima:
    void Jiggle()
    {
        using TensorFloat shift = ops.RandomUniform(new TensorShape(1, N, 3), -1f, 1f, (float)Time.time);
        TensorHelper.Replace(ref positions, ops.Add(positions, shift));
    }
    public void Go()
    {
        ribbon.gameObject.SetActive(true);
        s = startingS; //set the gradient descent interval
        DrawDistogram(distogram);
        Randomize();    
        go = true;
    }
    //The distance matrix cannot tell the difference between a protein and it's reflection so this is a useful function:
    public void ReflectProtein()
    {
        TensorHelper.Replace(ref positions, ops.Mad(positions, -1f, 0));
    }
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }
        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            s *= 2;
            Debug.Log($"Gradient step size: {s}");
        }
        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            s /= 2;
            Debug.Log($"Gradient step size: {s}");
        }
        if (Input.GetKeyDown(KeyCode.R))
        {
            ReflectProtein();
        }
        if (Input.GetKeyDown(KeyCode.Space))
        {
            Randomize();
        }
        if (Input.GetKeyDown(KeyCode.J))
        {
            Jiggle();
        }

        if (!go) return;

        DoFolding();
        Recenter();
        ribbon.UpdateRibbonMesh(protein, positions, N);
    }

    void DrawCurrentDistances()
    {
        //pseudo code: |x-x^T|/4

        using TensorFloat xT = ops.Transpose(positions, new int[] { 1, 0, 2 }) as TensorFloat;
        using TensorFloat DX = ops.Sub(positions, xT);
        using TensorFloat DX2 = ops.Mul(DX, DX);
        using TensorFloat XX = ops.ReduceSum(DX2, new int[] { 2 }, false);
        using TensorFloat X = ops.Sqrt(XX);
        using TensorFloat X_div = ops.Mul(X, 0.25f);
        using TensorFloat X0 = ops.Sub(1f, X_div);
        using TensorFloat T = X0.ShallowReshape(new TensorShape(1, 1, N, N)) as TensorFloat;
 
        (distancesImage.texture as RenderTexture)?.Release();
        distancesImage.texture = TextureConverter.ToTexture(T);
    }

    void Recenter()
    {
        using TensorFloat sumX = ops.ReduceSum(positions, new[] { 1 }, true);
        using TensorFloat midX = ops.Div(sumX, (float)N);
        TensorHelper.Replace(ref positions, ops.Sub(positions, midX));
    }


    private void OnApplicationQuit()
    {
        positions?.Dispose();
        ops?.Dispose();
        m_Allocator?.Dispose();
        distogram?.Dispose();
        worker?.Dispose();
    }

}
