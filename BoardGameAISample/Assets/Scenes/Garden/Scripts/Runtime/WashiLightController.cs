using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class WashiLightController : MonoBehaviour
{
    [SerializeField]
    private Vector3 size = new Vector3(4f, 2f, 0.5f);
    [SerializeField]
    private Material referenceMaterial;
    private Material _instancedMaterial;
    private Material instancedMaterial
    {
        get
        {
            if (_instancedMaterial == null)
            {
                _instancedMaterial = Instantiate(referenceMaterial);
                _instancedMaterial.name = $"{gameObject.name} {referenceMaterial.name}";
            }
            return _instancedMaterial;
        }
    }
    [SerializeField]
    private Texture2D texture = null;
    enum ChannelIndex { r, g, b, a };
    [SerializeField]
    private ChannelIndex channelIndex = ChannelIndex.r;
    [SerializeField]
    private float minIntensity = 0.5f;
    [SerializeField]
    private float maxIntensity = 2f;
    [SerializeField]
    private float randomDuration = 1f;
    [SerializeField]
    private float presenceRandomDuration = 5f;
    [SerializeField, Range(0f, 1f)]
    private float presenceAmount = 0.5f;

    [SerializeField]
    private string objectsSearchFilter = "WashiLight";
    [SerializeField]
    private Shader shaderSearchFilter;

    [SerializeField]
    private Renderer[] renderers;

    Vector3 worldU = Vector3.right;
    Vector3 worldV = Vector3.up;
    Vector4 tilingOffset = new Vector4(1, 1, 0, 0);

    [SerializeField]
    private bool displayAffectedRenderers = false;

    private float intensitySeed = 0f;
    private float presenceSeed = 0f;

    private void OnValidate()
    {
        size.x = Mathf.Abs(size.x);
        size.y = Mathf.Abs(size.y);
        size.z = Mathf.Abs(size.z);
    }

    private void Start()
    {
        ApplyTexture();

        intensitySeed = Random.value * 100f;
        presenceSeed = Random.value * 100f;
    }

    private void Update()
    {
        var intensity = Mathf.Lerp(minIntensity, maxIntensity, Mathf.PerlinNoise(Time.time / randomDuration, intensitySeed));
        var isHere = (presenceAmount >= 1)? true : Mathf.PerlinNoise(Time.time / presenceRandomDuration, presenceSeed) < presenceAmount;

        instancedMaterial.SetFloat("_Intensity", intensity);

        foreach (var r in renderers)
        {
            r.enabled = isHere;
        }
    }

    [ContextMenu("Apply Texture")]
    private void ApplyTexture()
    {
#if UNITY_EDITOR
        if (!Application.isPlaying)
            return;
#endif
        transform.localScale = Vector3.one;

        worldU = transform.right;
        worldV = transform.up;

        var start = transform.TransformPoint(-0.5f * size);
        var end = transform.TransformPoint(0.5f * size);

        var start2D = new Vector2(
            Vector3.Dot(start, worldU),
            Vector3.Dot(start, worldV)
            );

        var end2D = new Vector2(
            Vector3.Dot(end, worldU),
            Vector3.Dot(end, worldV)
            );

        tilingOffset = new Vector4(
            end2D.x - start2D.x,
            end2D.y - start2D.y,
            start2D.x,
            start2D.y
            );

        instancedMaterial.SetVector("_World_U", worldU);
        instancedMaterial.SetVector("_World_V", worldV);
        instancedMaterial.SetVector("_TilingOffset", tilingOffset);
        instancedMaterial.SetTexture("_Texture", texture);
        instancedMaterial.SetFloat("_Texture_Channel", (int)channelIndex);

        foreach (var r in renderers)
        {
            r.sharedMaterial = instancedMaterial;
        }
    }

    private void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.yellow;
        if (displayAffectedRenderers)
        {
            foreach (var r in renderers)
            {
                if (!r.enabled || !r.gameObject.activeSelf)
                    continue;

                var mesh = r.GetComponent<MeshFilter>()?.sharedMesh;
                if (mesh != null)
                {
                    Gizmos.matrix = r.transform.localToWorldMatrix;
                    Gizmos.DrawWireMesh(mesh);
                }
            }
        }

        Gizmos.matrix = Matrix4x4.TRS(transform.position, transform.rotation, Vector3.one);
        Gizmos.DrawWireCube(Vector3.zero, size);
    }

#if UNITY_EDITOR
    [ContextMenuItem("Get matching renderers in volume", "GetRenderers")]
    [ContextMenuItem("Get matching renderers in volume + enable", "GetAndEnableFacingRenderers")]
    private bool dummyField;

    void GetRenderers()
    {
        GetRenderersActive(false);
    }

    void GetRenderersActive(bool includeInactive)
    {
        transform.localScale = Vector3.one;

        var bounds = new Bounds(transform.position, Vector3.zero);
        float x = size.x * 0.5f;
        float y = size.y * 0.5f;
        float z = size.z * 0.5f;

        bounds.Encapsulate(transform.TransformPoint(new Vector3(x, y, z)));
        bounds.Encapsulate(transform.TransformPoint(new Vector3(-x, y, z)));
        bounds.Encapsulate(transform.TransformPoint(new Vector3(x, -y, z)));
        bounds.Encapsulate(transform.TransformPoint(new Vector3(-x, -y, z)));
        bounds.Encapsulate(transform.TransformPoint(new Vector3(x, y, -z)));
        bounds.Encapsulate(transform.TransformPoint(new Vector3(-x, y, -z)));
        bounds.Encapsulate(transform.TransformPoint(new Vector3(x, -y, -z)));
        bounds.Encapsulate(transform.TransformPoint(new Vector3(-x, -y, -z)));

        renderers = FindObjectsOfType<Renderer>(includeInactive).Where(r => {
            return r.bounds.Intersects(bounds) && r.gameObject.name.Contains(objectsSearchFilter) && ((shaderSearchFilter == null) ? true : r.sharedMaterials.Any(m => m.shader == shaderSearchFilter));
        }).ToArray();
    }

    void GetAndEnableFacingRenderers()
    {
        GetRenderersActive(true);

        foreach (var r in renderers)
            r.gameObject.SetActive( Vector3.Dot(r.transform.forward, transform.forward) < 0 );

        renderers = renderers.Where(r => r.gameObject.activeSelf).ToArray();
    }
#endif
}
