using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;
using System.Threading.Tasks;


/*
 * This is the main class which loads the features of the protein then applies the Alphafold model in
 * multiple windows (crops) to guess the distance matrix of the protein. The distance matrix gives the distances
 * between any two amnio acids on the protein. 
 * 
 * This simplified implementation of Alphafold v1 is for demonstration purposes only. For the latest protein
 * folding models see for example Alphafold v2, Omegafold or ESMfold.
 */
public class Main : MonoBehaviour
{
    public ModelAsset onnx;
    IWorker worker;
    static Ops ops;
    ITensorAllocator m_Allocator;

    public Text sequenceText, instructionText;
    public Canvas controlPanel;
    public GameObject buttonPanel;

    public FoldingProtein proteinFolder;

    enum Mode { START, LOADING_PROTEIN, FOLDING };
    Mode mode = Mode.START;

    public static Features featureStruct;

    string sequence = "";

    public const int WINDOW_SIZE = 64; // The size of the window looking at a sub-sequence of amnio acids. Must be 64

    const int STRIDE = 48; // How far the crop window moves each time. Must be less than the window size. Can be changed

    // The starting positions of the crop windows
    int[] startsX = new int[] { 0 };
    int[] startsY = new int[] { 0 };

    int currentProteinID = 0;

    void Start()
    {
        controlPanel.gameObject.SetActive(false);

        Model model = ModelLoader.Load(onnx);
        worker = model.CreateWorker(Unity.Sentis.DeviceType.GPU);

        m_Allocator = new TensorCachingAllocator();
        ops = WorkerFactory.CreateOps(BackendType.GPUCompute, m_Allocator);

        sequenceText.text = "";
        LoadFeatures.GetProteinList();
    }

    public void HomeButtonClicked()
    {
        instructionText.text = "Choose one of the protein feature sets.";
        proteinFolder.Stop();
        controlPanel.gameObject.SetActive(false);
        buttonPanel.SetActive(true);
        sequenceText.text = "";
    }

    // Load the features of the chosen molecule then run the Alphafold model
    public async void OnChooseMolecule(int ID)
    {
        if (mode == Mode.LOADING_PROTEIN) return;
        mode = Mode.LOADING_PROTEIN;
        controlPanel.gameObject.SetActive(false);
        sequenceText.text = "";
        currentProteinID = ID;
        buttonPanel.SetActive(false);

        instructionText.text = $"Loading features of protein {LoadFeatures.proteinNames[ID]} ...";
        float startTime = Time.time;
        proteinFolder.Stop();

        featureStruct = await Task.Run(() => LoadFeatures.Load(ID));

        // wait until at least 2 seconds passed so we have time to read the text
        await Task.Delay((int)Mathf.Max(0, 2000 - (Time.time - startTime) * 1000));
        FoldingProtein.N = (int)featureStruct.seq_length[0][0];
        // aatype to IDs
        featureStruct.aminoAcidIDs = new int[FoldingProtein.N];
        for (int i = 0; i < FoldingProtein.N; i++)
        {
            featureStruct.aminoAcidIDs[i] = System.Array.IndexOf(featureStruct.aatype[i], 1f);
        }
        sequence = featureStruct.sequence;
        sequenceText.text = $"Amino Acid Sequence of {LoadFeatures.proteinNames[currentProteinID]}:\n{sequence} ({FoldingProtein.N})";

        controlPanel.gameObject.SetActive(true);
        instructionText.text = "Running Alphafold v1 model...";
        FoldingProtein.go = false;
        StartCoroutine(RunAphafoldModel());
    }


    public IEnumerator<int> RunAphafoldModel()
    {
        int N = FoldingProtein.N;

        // We make sure to dispose these at the end:
        TensorFloat distogramProbs = new TensorFloat(new TensorShape(1, N, N, WINDOW_SIZE), new float[N * N * WINDOW_SIZE]);
        TensorFloat weightsTensor = new TensorFloat(new TensorShape(1, N, N, 1), new float[N * N]);

        using TensorFloat distogramTensor = new TensorFloat(new TensorShape(1, N, N, 1), new float[N * N]);
        using TensorFloat ones = ops.ConstantOfShape(new TensorShape(new TensorShape(1, WINDOW_SIZE, WINDOW_SIZE, 1)), 1f);

        // calculate how many step we will need to cover whole sequence
        int steps = Mathf.FloorToInt((N - WINDOW_SIZE - 1) * 1f / STRIDE) + 2;

        // store the stating position of each window
        startsX = new int[steps];
        startsY = new int[steps];
        for (int i = 0; i < startsX.Length; i++)
        {
            startsX[i] = Mathf.Min(i * STRIDE, N - WINDOW_SIZE);
            startsY[i] = Mathf.Min(i * STRIDE, N - WINDOW_SIZE);
        }

        // we need this to calculate the mean distance. An array of distances: 0..63
        using TensorFloat distRange = ops.Range(0, (float)64, 1f).ShallowReshape(new TensorShape(1, 1, 1, 64)) as TensorFloat;

        foreach (int startx in startsX)
        {
            foreach (int starty in startsY)
            {
                using TensorFloat features = LoadFeatures.CreateCrop(ops, startx, starty, featureStruct);

                RunNetwork(N, features, ref distogramProbs, ref weightsTensor, ones, distRange, startx, starty);

                PreviewDistogram(distogramProbs, weightsTensor, distRange);

                //we yield here to allow the graphics to update
                yield return 0;
            }
        }

        // symmetrise the probabilities (since distance AB = distance BA).
        using TensorFloat distProbTrans = ops.Transpose(distogramProbs, new int[] { 0, 2, 1, 3 }) as TensorFloat;
        using TensorFloat distSym = ops.Add(distogramProbs, distProbTrans);
        using TensorFloat d3 = ops.Div(distSym, 2f) as TensorFloat;
        using TensorFloat d4 = ops.Div(d3, weightsTensor);

        using TensorFloat a2 = ops.Mul(distRange, d4);
        using TensorFloat likelyDist = ops.ReduceSum(a2, new int[] { 3 }, false);

        using TensorFloat l1 = ops.Div(likelyDist, 63f);

        using TensorFloat l2 = ops.Mul(l1, 6f);
        using TensorFloat l2_squared = ops.Square(l2);

        proteinFolder.UpdateDistogramSquared( l2_squared.ShallowReshape(new TensorShape(N, N)) as TensorFloat );
        proteinFolder.UpdateDistogram( l1.ShallowReshape(new TensorShape(N, N)) as TensorFloat);

        distogramProbs.Dispose();
        weightsTensor.Dispose();

        instructionText.text = "";
        mode = Mode.FOLDING;

        proteinFolder.Go();
    }

    // Run the network over the cropped window of features
    private void RunNetwork(int N, TensorFloat features, ref TensorFloat distogramProbs, ref TensorFloat weightsTensor, TensorFloat ones, TensorFloat range, int startx, int starty)
    {
        using TensorInt cropx = new TensorInt(new TensorShape(1, 2), new[] { startx, startx + WINDOW_SIZE });
        using TensorInt cropy = new TensorInt(new TensorShape(1, 2), new[] { starty, starty + WINDOW_SIZE });

        var inputs = new Dictionary<string, Tensor> { { "input", features }, { "cropx", cropx }, { "cropy", cropy } };

        worker.Execute(inputs);

        var distanceProbs = worker.PeekOutput("distance_probs") as TensorFloat; // shape=(1, WINDOW_SIZE, WINDOW_SIZE, 64)

        int[] padding = GetPadding(N, startx,starty);

        using TensorFloat paddedDist = ops.Pad(distanceProbs, padding);
        using TensorFloat paddedOnes = ops.Pad(ones, padding);

        // Add results to total:
        TensorHelper.Replace( ref distogramProbs, ops.Sum(new[] { distogramProbs, paddedDist }));
        TensorHelper.Replace( ref weightsTensor , ops.Sum(new[] { weightsTensor, paddedOnes }));
    }

    // Calculate the padding from the cropped window to the full distogram
    int[] GetPadding(int N, int startx,int starty)
    {
        int padL = startx;
        int padR = N - (startx + WINDOW_SIZE);
        int padU = starty;
        int padD = N - (starty + WINDOW_SIZE);
        return new int[] { 0, padU, padL, 0, 0, padD, padR, 0 };
    }

    void PreviewDistogram(TensorFloat distogramProbs, TensorFloat weightsTensor, TensorFloat range)
    {
        using TensorFloat c = ops.Div(distogramProbs, weightsTensor);
        using TensorFloat b = ops.Mul(range, c);
        using TensorFloat distSoFar = ops.ReduceSum(b, new[] { 3 }, false);
        using TensorFloat distSoFarNorm = ops.Div(distSoFar, 63f);
        // draw
        proteinFolder.DrawDistogram(distSoFarNorm);
    }

    void Update()
    {
        if (Input.anyKeyDown)
        {
            for (int keycode = (int)KeyCode.Alpha0; keycode <= (int)KeyCode.Alpha9; keycode++)
            {
                if (Input.GetKeyDown((KeyCode)keycode)) OnChooseMolecule(keycode - (int)KeyCode.Alpha0);
            }
            for (int keycode = (int)KeyCode.Keypad0; keycode <= (int)KeyCode.Keypad9; keycode++)
            {
                if (Input.GetKeyDown((KeyCode)keycode)) OnChooseMolecule(keycode - (int)KeyCode.Keypad0);
            }
        }
    }

    private void OnApplicationQuit()
    {
        worker?.Dispose();
        ops?.Dispose();
        m_Allocator?.Dispose();
    }

}
