using System.Collections.Generic;
using UnityEngine;
using Newtonsoft.Json;
using System.IO;
using Unity.Sentis;

/*
 * This is the feature struct that contains all the given information about a particular protein sequence
 * It contains information about this particular squence as well as how to it compares to other protein
 * sequences from the database. The idea being that correlations between pairs of amino acids in different
 * proteins give clues as to which parts of a protein may be connected.
 * 
 * The features are loaded from JSON files. The features were prepared by Deep Mind based on protein sequences
 * provided by the CASP 13 competition.
 * 
 * We provided 10 examples of protein feature files. Press 0..9 to load a different one.
 * 
 * For more information see: https://github.com/Urinx/alphafold_pytorch
 * and https://github.com/google-deepmind/deepmind-research/tree/master/alphafold_casp13
 */
public struct Features
{
    // 2D structures giving information about relationships to other proteins
    // The number on the right gives the size of the rank-3 index
    public float[][][] pseudo_frob;//                                       1 
    public float[][][] pseudolikelihood;//                                  484
    public float[][][] gap_matrix;//                                        1
    
    //1D structures
    public float[][] aatype; //amino acid types / one hot encoding          21

    public float[][] deletion_probability;//                                1
    public float[][] hhblits_profile;//                                     22
    public float[][] hmm_profile;//                                         30

    public float[][] non_gapped_profile;//                                  21
    public float[][] num_alignments;//                                      1

    public float[][] profile;//                                             21
    public float[][] profile_with_prior;//                                  22
    public float[][] profile_with_prior_without_gaps;//                     21
    public float[][] pseudo_bias;  //                                       22

    public float[][] residue_index; //                                      1
    public float[][] reweighted_profile;//                                  22
    public float[][] sec_structure;//                                       8
    public float[][] sec_structure_mask;//                                  1
    public float[][] seq_length;//                                          1

    //some string information
    public string domain_name;
    public string chain_name;
    public string superfamily;
    public string sequence;

    public int[] aminoAcidIDs;
}

public class LoadFeatures 
{
    public static string JSONfolder = Application.streamingAssetsPath + @"/Protein Features/";
    public static string statsFile = Application.streamingAssetsPath + @"/Stats/Stats_train_s35.json";
    
    static dynamic stats;
    public static string[] proteinFiles;
    public static List<string> proteinNames = new List<string>();

    public static Features Load(int ID)
    {
        string json = File.ReadAllText(proteinFiles[ID]);
        Features featureStruct = JsonConvert.DeserializeObject<Features>(json);
        string jsonStats = File.ReadAllText(statsFile);
        stats = JsonConvert.DeserializeObject(jsonStats);
        return featureStruct;
    }

    // Creates the sliding input window from the features. This is what is fed into the neural network model
    // There are 1876 layers of features
    public static TensorFloat CreateCrop(Ops ops, int startx, int starty, Features f)
    {
        List<TensorFloat> featureList = new List<TensorFloat>();

        featureList.Add(Feature2D(startx, starty, f.pseudo_frob, mean: (float)stats.mean.pseudo_frob, vari: (float)stats.var.pseudo_frob));

        // layers: 1..1458 = 3x(484+2) pseudo_frob full matrix (We skip this input out for this simple example).
        // We get fairly good results without it for this demonstration so we'll just set these to zero:
        featureList.Add(ops.ConstantOfShape(new TensorShape(1, 1458, Main.WINDOW_SIZE, Main.WINDOW_SIZE), 0f));

        // These are supposed to help with the model judging distances between amino acids
        // They don't make a big difference and could be set to zero
        featureList.Add(PositionalFeatures(ops, startx, starty));

        // Bit coding is another way to help the model recognise distances (layers 1460...1466):
        // Basically just stripes of various sizes
        for (int z = 0; z < 2; z++)
        {
            for (int k = 0; k < 4; k++)
            {
                featureList.Add(BitCodes(startx, starty, 1 << k, isHorizontal: (z == 1)));
            }
        }

        // layer 1468 onwards
        // We add these 1D arrays twice. Once vertically and once horizontally.
        // They are normalized by mean and variation as given in the json file to put them in the correct range that was used for training
        // We've labelled the most important ones which seem to have the biggest effect on the protein shape
        for (int z = 0; z < 2; z++)
        {
            bool isHorizontal = (z == 1);
            featureList.Add(Feature1D(startx, starty, f.profile, isHorizontal, mean: (float)stats.mean.profile, vari: (float)stats.var.profile));// (important)  
            featureList.Add(Feature1D(startx, starty, f.hhblits_profile, isHorizontal, mean: (float)stats.mean.hhblits_profile, vari: (float)stats.var.hhblits_profile));
            featureList.Add(Feature1D(startx, starty, f.aatype, isHorizontal));
            featureList.Add(Feature1D(startx, starty, f.deletion_probability, isHorizontal, mean: (float)stats.mean.deletion_probability, vari: (float)stats.var.deletion_probability));
            featureList.Add(Feature1D(startx, starty, f.pseudo_bias, isHorizontal, mean: (float)stats.mean.pseudo_bias, vari: (float)stats.var.pseudo_bias));
            featureList.Add(Feature1D(startx, starty, f.profile_with_prior, isHorizontal, mean: (float)stats.mean.profile_with_prior, vari: (float)stats.var.profile_with_prior));
            featureList.Add(Feature1D(startx, starty, f.profile_with_prior_without_gaps, isHorizontal, mean: (float)stats.mean.profile_with_prior_without_gaps, vari: (float)stats.var.profile_with_prior_without_gaps));
            featureList.Add(Feature1D(startx, starty, f.reweighted_profile, isHorizontal, mean: (float)stats.mean.reweighted_profile, vari: (float)stats.var.reweighted_profile));
            featureList.Add(Feature1D(startx, starty, f.non_gapped_profile, isHorizontal, mean: (float)stats.mean.non_gapped_profile, vari: (float)stats.var.non_gapped_profile)); //  (important) 
            featureList.Add(Feature1D(startx, starty, f.hmm_profile, isHorizontal, mean: (float)stats.mean.hmm_profile, vari: (float)stats.var.hmm_profile)); //  (important) 
            featureList.Add(Feature1D(startx, starty, f.num_alignments, isHorizontal, mean: (float)stats.mean.num_alignments, vari: (float)stats.var.num_alignments));
            featureList.Add(Feature1D(startx, starty, f.seq_length, isHorizontal, mean: (float)stats.mean.seq_length, vari: (float)stats.var.seq_length));
        }
        var result = ops.Concat(featureList.ToArray(), 1) as TensorFloat;
        foreach(var feature in featureList)
        {
            feature.Dispose();
        }
        return result; // shape should be (1, 1878, 64, 64)
    }

    // Here we add the 1D array of features which are tiled either vertically or horizontally
    static TensorFloat Feature1D(int startX, int startY, float[][] f, bool vertical, float mean = 0f, float vari = 1f)
    {
        int S = Main.WINDOW_SIZE;
        int layers = f[0].Length;
        float[] res = new float[layers * S * S];

        float std = Mathf.Sqrt((float)vari);
        if (std < 1e-12) std = 1f;

        for (int i = 0; i < layers; i++)
        {
            for (int y = 0; y < S; y++)
            {
                for (int x = 0; x < S; x++)
                {
                    float val = vertical ? f[y + startY][i] : f[x + startX][i];
                    val = (val - mean) / std;
                    res[S * S * i + S * y + x] = val;
                }
            }
        }
        return new TensorFloat(new TensorShape(1, layers, S, S), res);
    }

    static TensorFloat Feature2D(int startx, int starty, float[][][] f, float mean, float vari)
    {
        int S = Main.WINDOW_SIZE;
        float[] res = new float[S * S];
        float std = Mathf.Sqrt(vari);      
        for (int x = 0; x < S; x++)
        {
            for (int y = 0; y < S; y++)
            {
                res[y * S + x] = (f[y + starty][x + startx][0] - mean) / std;
            }
        }
        return new TensorFloat(new TensorShape(1, 1, S, S), res);
    }

    // This is basically given by the formula: (x-y)/100
    static TensorFloat PositionalFeatures(Ops ops, int startx, int starty)
    {
        int S = Main.WINDOW_SIZE;
        using TensorInt Xrange = ops.Range(startx, startx + S, 1);
        using TensorInt Xrange2 = Xrange.ShallowReshape(new TensorShape(1, 1, 1, S)) as TensorInt;
        using TensorInt Yrange = ops.Range(starty, starty + S, 1);
        using TensorInt Yrange2 = Yrange.ShallowReshape(new TensorShape(1, 1, S, 1)) as TensorInt; ;
        using TensorInt diff = ops.Sub(Xrange2, Yrange2);
        using TensorFloat diffF = ops.Cast(diff, DataType.Float) as TensorFloat;
        return ops.Div(diffF, 100f);
    }

    //This could also be done using tensors but it is non critical:
    static TensorFloat BitCodes(int startx, int starty, int size, bool isHorizontal=true)
    {      
        int S = Main.WINDOW_SIZE;
        float[] res = new float[S * S];
        for (int x = 0; x < S; x++)
        {
            for (int y = 0; y < S; y++)
            {
                (int x2, int y2) = (x + startx, y + starty);
                float bit = isHorizontal ? (y2 / size) % 2 : (x2 / size) % 2;
                res[y * S + x] = bit;
            }
        }
        return new TensorFloat(new TensorShape(1, 1, S, S), res);
    }

    public static void GetProteinList()
    {
        string[] files = Directory.GetFiles(LoadFeatures.JSONfolder);
        var fileList = new List<string>();
        foreach (string file in files)
        {
            string[] parts = Path.GetFileName(file).Split(".");

            if (parts[parts.Length - 1] == "json")
            {
                proteinNames.Add(parts[0]);
                fileList.Add(file);
            }
        }
        proteinFiles = fileList.ToArray();
    }
}
