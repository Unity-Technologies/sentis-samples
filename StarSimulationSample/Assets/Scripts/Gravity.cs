using UnityEngine;
using Unity.Sentis;

public class Gravity : MonoBehaviour
{
    //public ModelAsset modelAsset;
    IWorker worker;
    GPUComputeBackend backend;
    
    public GameObject planet;
    GameObject[] planets = new GameObject[N];
    const int N = 3000;

    const float t = 0.001f; // TimeStep
    const float G = -9f; // gravity

   TensorFloat x = TensorFloat.AllocZeros(new TensorShape(1, N, 3));  // positions
   TensorFloat p = TensorFloat.AllocZeros(new TensorShape(N, 3)); // momentum (= mass * velocity)
   TensorFloat m = TensorFloat.AllocZeros(new TensorShape(N)); // mass

    void Start()
    {
        backend = new GPUComputeBackend();

        float massMin = 0.0001f;
        float massMax = 100f;

        // Randomize positions, momenta and masses
        for (int i = 0; i < N; i++)
        {

            Vector3 pos = UnityEngine.Random.insideUnitSphere;
            x[i, 0] = pos.x * 10;
            x[i, 1] = pos.y;
            x[i, 2] = pos.z * 10;

            float mass = Mathf.Exp(UnityEngine.Random.Range(-1f, 1f));
            m[i] = mass;

            p[i, 0] = 10f * mass * 1.0f;
            p[i, 1] = 10f * mass * 0.0f;
            p[i, 2] = 10f * mass * 1.0f;

            float radius = Mathf.Pow(mass, 1.0f / 3.0f) / (4f * 3.14f); // m ~ 4Pir^3

            planets[i] = Instantiate(planet);
            planets[i].transform.position = new Vector3(x[i, 0], x[i, 1], x[i, 2]);
            planets[i].transform.localScale =  Vector3.one * Mathf.Clamp(0.1f*radius, 0.02f, 0.1f);
            Renderer renderer = planets[i].GetComponent<Renderer>();
            renderer.material.SetFloat("MassRatio", (mass - massMin)/(massMax - massMin));
        }


        var xdef = new InputDef(DataType.Float, x.shape);
        var pdef = new InputDef(DataType.Float, p.shape);
        var mdef = new InputDef(DataType.Float, m.shape);

        var modelMotion = Functional.Compile((x, m, p) =>
        {
            // https://en.wikipedia.org/wiki/N-body_problem
            // Hamilton equation of motion
            // position q_i
            // momentum p_i = m_i dq_i/dt
            // dq_i/dt = dH/dp_i dp_i/dt = -dH/dq_i
            // H = T + U = Sum_i(||p_i|| ^ 2 / 2m_i) - Sum_i,j(G * m_j * m_i / ||q_j - q_i||)
            // write this in tensor form

            var xT = Functional.Transpose(x, 0, 1);
            // Use this as to avoid 0/0 when i=j
            var epsilon = 0.000001f;

            // create array of distance vectors
            var xy = x - xT;

            // calculate r^2 distances
            var R2 = Functional.ReduceSum(xy * xy, -1, false) + epsilon;
            R2 = Functional.Unsqueeze(R2, 2);

            // multiply by masses
            var mu1 = Functional.Unsqueeze(m, 1);
            var mu = Functional.Unsqueeze(mu1, 1);
            var Mxy = mu * xy;

            var a = Functional.ReduceSum(Mxy / Functional.Pow(R2, 1.5f), 0);

            var dp = mu1 * a;

            // p_n+1 = p_n + G * t * dp
            // x_n+1 = x_n + p_n / m) * t
            return (p + (G * t * dp), x + (p / Functional.Unsqueeze(m, 1)) * t);
        },
          (xdef, mdef, pdef)
        );

        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, modelMotion);
    }

    void Update()
    {
        worker.SetInput("input_0", x);
        worker.SetInput("input_1", m);
        worker.SetInput("input_2", p);
        worker.Execute();

        var p_n = worker.PeekOutput("output_0") as TensorFloat;
        var x_n = worker.PeekOutput("output_1") as TensorFloat;

        backend.MemCopy(p_n, p);
        backend.MemCopy(x_n, x);

        for (int i = 0; i < N; i++)
        {
            Renderer renderer = planets[i].GetComponent<Renderer>();
            renderer.material.SetBuffer("Positions", ComputeTensorData.Pin(x, clearOnInit: false).buffer);
            renderer.material.SetInt("Index", i);
        }
    }

    private void OnDestroy()
    {
        x.Dispose();
        p.Dispose();
        m.Dispose();
        backend.Dispose();
        worker.Dispose();
    }
}
