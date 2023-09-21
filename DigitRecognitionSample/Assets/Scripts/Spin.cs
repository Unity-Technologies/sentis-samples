using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/*
 *  Rotate objecst along the upwards axis
 */
public class Spin : MonoBehaviour
{
    public float rotationSpeed = 1;

    void Update()
    {
        transform.Rotate(Vector3.up, 360 * Time.deltaTime * rotationSpeed);
    }
}
