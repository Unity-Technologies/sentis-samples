using UnityEngine;
using Unity.Sentis;

public class Gravity : MonoBehaviour
{
    Worker worker;

    public GameObject planet;
    GameObject[] planets = new GameObject[N];
    const int N = 3000;

    const float t = 0.001f; // TimeStep
    const float G = -9f; // gravity

    Tensor<float> x;
    Tensor<float> p;
    Tensor<float> m;

    void Start()
    {
        float massMin = 0.0001f;
        float massMax = 100f;
        
        x = new Tensor<float>(new TensorShape(1, N, 3)); // positions
        p = new Tensor<float>(new TensorShape(N, 3)); // momentum (= mass * velocity)
        m = new Tensor<float>(new TensorShape(N)); // mass

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

        var graph = new FunctionalGraph();
        var xdef = graph.AddInput<float>(x.shape);
        var mdef = graph.AddInput<float>(m.shape);
        var pdef = graph.AddInput<float>(p.shape);

        // https://en.wikipedia.org/wiki/N-body_problem
        // Hamilton equation of motion
        // position q_i
        // momentum p_i = m_i dq_i/dt
        // dq_i/dt = dH/dp_i dp_i/dt = -dH/dq_i
        // H = T + U = Sum_i(||p_i|| ^ 2 / 2m_i) - Sum_i,j(G * m_j * m_i / ||q_j - q_i||)
        // write this in tensor form

        var xT = Functional.Transpose(xdef, 0, 1);

        // Use this as to avoid 0/0 when i=j
        var epsilon = 0.000001f;

        // create array of distance vectors
        var xy = xdef - xT;

        // calculate r^2 distances
        var R2 = Functional.ReduceSum(xy * xy, -1, false) + epsilon;
        R2 = Functional.Unsqueeze(R2, 2);

        // multiply by masses
        var mu1 = Functional.Unsqueeze(mdef, 1);
        var mu = Functional.Unsqueeze(mu1, 1);
        var Mxy = mu * xy;

        var a = Functional.ReduceSum(Mxy / Functional.Pow(R2, 1.5f), 0);

        var dp = mu1 * a;

        // p_n+1 = p_n + G * t * dp
        // x_n+1 = x_n + (p_n / m) * t
        var output1 = pdef + (G * t * dp);
        var output2 = xdef + (pdef / Functional.Unsqueeze(mdef, 1)) * t;

        var modelMotion = graph.Compile(output1, output2);

        worker = new Worker(modelMotion, BackendType.GPUCompute);
    }

    void Update()
    {
        worker.Schedule(x, m, p);

        p?.Dispose();
        x?.Dispose();
        p = (worker.PeekOutput("output_0") as Tensor<float>).ReadbackAndClone();
        x = (worker.PeekOutput("output_1") as Tensor<float>).ReadbackAndClone();

        for (int i = 0; i < N; i++)
        {
            Renderer renderer = planets[i].GetComponent<Renderer>();
            renderer.material.SetBuffer("Positions", ComputeTensorData.Pin(x, clearOnInit: false).buffer);
            renderer.material.SetInteger("Index", i);
        }
    }

    void OnDestroy()
    {
        x?.Dispose();
        p?.Dispose();
        m?.Dispose();
        worker?.Dispose();
    }
}
