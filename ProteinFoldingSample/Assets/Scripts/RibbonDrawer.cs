using UnityEngine;
using Unity.Sentis;
using System.Threading.Tasks;

// This class is responsible for drawing the ribbon representing the protein and also controls rotating it.
public class RibbonDrawer: MonoBehaviour 
{
    // arrays for ribbon mesh
    Vector3[] normals = new Vector3[1];
    Vector3[] vertices = new Vector3[1];
    Color[] colors = new Color[1];
    Vector3[] tangents = new Vector3[1];
    Vector3[] ups = new Vector3[1];
    int[] tris = new int[1];

    const float tubeRadius = 0.2f;
    const float tubeAspectRatio = 0.1f;//0.25f;//1f; //change this to 1f to make tubes

    bool drag = false;

    Vector3 rotationAxis = Vector3.up * 10f;
    const float mouseSensititvity = 200f;
    enum ColorScheme { RAINBOW, AMINOACID, BLUE };
    ColorScheme colorScheme = ColorScheme.RAINBOW;

    // Updates the ribbon mesh. The sections are the number of sections of tube between each amino acid.
    // Circ is the number of points on the circumference of each circle around the tube.
    // The orientation of the ribbon is merely calculated from the embedding of the curve and not
    // meant to represent any chemical properties of the protein. Although it does highlight the sheets
    // and helix structures as a consequence. 
    public void UpdateRibbonMesh(GameObject protein, TensorFloat xTensor, int N, int sections = 8, int circ = 12)
    {
        Vector3[] aminoPos = TensorHelper.GetVectorArray(xTensor);

        // Calculate number of vertices and number of triangles
        int numVerts = circ * ((N - 1) * sections + 1);
        int numTris = (N - 1) * circ * sections * 6;

        Mesh mesh = protein.GetComponent<MeshFilter>().mesh;
        // Initialize the arrays
        if (vertices.Length != numVerts)
        {
            vertices = new Vector3[numVerts];
            Destroy(mesh);
            mesh = protein.GetComponent<MeshFilter>().mesh = new Mesh();
        }
        if (normals.Length != numVerts) normals = new Vector3[numVerts];
        if (colors.Length != numVerts) colors = new Color[numVerts];
        if (tangents.Length != N) tangents = new Vector3[N];
        if (tris.Length != numTris) tris = new int[numTris];
        if (ups.Length != N) ups = new Vector3[N];

        // calculate an array of vectors that will be perpendicular to the ribbon surface:
        for (int i = 1; i < N - 1; i++)
        {
            ups[i] = Vector3.Cross(aminoPos[i - 1] - aminoPos[i], aminoPos[i + 1] - aminoPos[i]).normalized;
            //avoid 180 degree twists:
            if (Vector3.Dot(ups[i], ups[i - 1]) < 0) ups[i] *= (-1);
        }
        int t = circ * sections;

        // Calculate the ribbon vertices and normals for the mesh
        // Speed this up by spreading over all CPUs
        Parallel.For(0, N, i =>
        {
            Vector3 midPoint = aminoPos[i];
            Vector3 up = ups[i];
            Vector3 left = Vector3.zero;
            int numSlices = i < N - 1 ? sections : 1;
            for (int c = 0; c < circ; c++)
            {
                float b = 2 * c * Mathf.PI / circ;
                float CS = Mathf.Cos(b);
                float SN = Mathf.Sin(b);
                for (int s = 0; s < numSlices; s++)
                {
                    float a = s * 1f / sections;

                    if (i < N - 1)
                    {
                        Vector3 C1 = i == 0 ? aminoPos[i] : aminoPos[i] + (aminoPos[i + 1] - aminoPos[i - 1]).normalized * 0.5f;
                        Vector3 C2 = i >= N - 2 ? aminoPos[i + 1] : aminoPos[i + 1] - (aminoPos[i + 2] - aminoPos[i]).normalized * 0.5f;

                        midPoint = VectorHelper.Bezier(aminoPos[i], aminoPos[i + 1], C1, C2, a);

                        Vector3 tangent = VectorHelper.BezierTangent(aminoPos[i], aminoPos[i + 1], C1, C2, a);

                        // This calculates a reasonably smooth interpolation of the ribbon
                        Quaternion rot = Quaternion.identity;
                        if (tangent.sqrMagnitude > 0)
                            rot.SetLookRotation(tangent, ups[i] * (1 - a) + ups[i + 1] * a);
                        up = rot * Vector3.up;
                        left = rot * Vector3.left;
                    }

                    Vector3 norm = (tubeAspectRatio * CS * up + SN * left).normalized;
                    Vector3 tubeOffset = CS * up + tubeAspectRatio * SN * left;
                    //end caps
                    if (i == 0 && s == 0)
                    {
                        tubeOffset = Vector3.zero;
                        norm = (aminoPos[0] - aminoPos[1]).normalized;
                    }
                    if (i == N - 1 && s == 0)
                    {
                        tubeOffset = Vector3.zero;
                        norm = (aminoPos[N - 1] - aminoPos[N - 2]).normalized;
                    }

                    int index = t * i + s * circ + c;
                    vertices[index] = midPoint + tubeOffset * tubeRadius;
                    normals[index] = norm;
                    colors[index] = GetColor(i + a, N);
                }
            }
        }
        );

        // Set the triangles       
        for (int i = 0; i < (N - 1) * sections; i++)
        {
            for (int j = 0; j < circ; j++)
            {
                int K = circ * 6 * i + (j % circ) * 6;
                int i2 = i + 1;
                int j2 = (j + 1) % circ;

                tris[K + 0] = circ * i + j;
                tris[K + 1] = circ * i + j2;
                tris[K + 2] = circ * i2 + j;

                tris[K + 3] = circ * i2 + j;
                tris[K + 4] = circ * i + j2;
                tris[K + 5] = circ * i2 + j2;
            }
        }

        // set the mesh data
        mesh.vertices = vertices;
        mesh.triangles = tris;
        mesh.colors = colors;
        mesh.normals = normals;
    }

    // Control the rotating of the protein
    private void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            drag = true;
        }
        if (Input.GetMouseButtonUp(0))
        {
            drag = false;
        }
        Vector2 mouseMove = new Vector2(Input.GetAxis("Mouse X"), Input.GetAxis("Mouse Y"));
        if (drag)
        {
            Vector3 axis = new Vector3(mouseMove.y, -mouseMove.x, 0) * mouseSensititvity;
            rotationAxis = 0.9f * rotationAxis + 0.1f * (axis);
        }
        float mouseZ = Input.GetAxis("Mouse ScrollWheel");
        Camera.main.transform.position += new Vector3(0, 0, mouseZ);

        transform.Rotate(rotationAxis, Time.deltaTime * rotationAxis.magnitude, Space.World);
    }

    public Color GetColor(float i, int N)
    {
        switch (colorScheme)
        {
            case ColorScheme.AMINOACID:
                int ID = Main.featureStruct.aminoAcidIDs[Mathf.FloorToInt(i + 0.5f)];
                return AminoAcids.colorRasmol[ID];
            case ColorScheme.BLUE:
                return new Color(0, 0.5f, 1f);
            default:
            case ColorScheme.RAINBOW:
                return Color.HSVToRGB((1 - i * 1f / N) * 0.55f, 1, 1);
        }
    }
}
