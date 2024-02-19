using UnityEngine;

// Helper class with information about the different Amino Acids and the Rasmol color scheme.
public class AminoAcids
{
    public static string letters = "ARNDCQEGHILKMFPSTWYVX-";
    public static string[] shortnames = "ala, arg, asn, asp, cys, gln, glu, gly, his, ile, leu, lys, met, phe, pro, ser, thr, trp, tyr, val, xaa, gap".Split(',');
    public static Color[] colorRasmol = new Color[22]
    {
        new Color32(200, 200, 200, 255),
        new Color32(20, 90, 255, 255),
        new Color32(0, 220, 220, 255),
        new Color32(230, 10, 10, 255),
        new Color32(230, 230, 0, 255),
        new Color32(0, 220, 200, 255),
        new Color32(230, 230, 0, 255),
        new Color32(235, 235, 235, 255),
        new Color32(130, 130, 210, 255),
        new Color32(15, 130, 15, 255),
        new Color32(15, 130, 15, 255),
        new Color32(20, 90, 255, 255),
        new Color32(230, 230, 0, 255),
        new Color32(50, 50, 170, 255),
        new Color32(220, 150, 130, 255),
        new Color32(250, 150, 0, 255),
        new Color32(250, 150, 0, 255),
        new Color32(180, 90, 180, 255),
        new Color32(50, 50, 170, 255),
        new Color32(15, 130, 15, 255),
        new Color32(190, 160, 110, 255),
        Color.black
    };
}
