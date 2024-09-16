using System;
using UnityEngine;

public class ImagePreview : MonoBehaviour
{
    public GameObject imageQuad;

    public void SetTexture(Texture texture)
    {
        imageQuad.GetComponent<MeshRenderer>().material.mainTexture = texture;
        var aspectRatio = texture.width / (float)texture.height;
        imageQuad.transform.localScale = new Vector3(aspectRatio, 1f, 1f);
    }
}
