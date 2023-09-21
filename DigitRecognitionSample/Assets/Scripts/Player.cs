using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/*
 *  Handles player
 *   - player being able to push objects.
 *   - detects when it hits a rigidbody gives it a bit of velocity
 */
public class Player : MonoBehaviour
{
    // this script pushes all rigid bodies that the character touches
    float pushPower = 2.0f;
    void OnControllerColliderHit(ControllerColliderHit hit)
    {
        Rigidbody body = hit.collider.attachedRigidbody;
        if (body == null || body.isKinematic)
        {
            return;
        }
        // We dont want to push objects below us
        if (hit.moveDirection.y < -0.3)
        {
            return;
        }

        Vector3 pushDir = new Vector3(hit.moveDirection.x, 0, hit.moveDirection.z);
        body.velocity = pushDir * pushPower;
    }
}
