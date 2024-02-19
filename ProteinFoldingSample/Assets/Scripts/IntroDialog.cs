using UnityEngine;

public class IntroDialog : MonoBehaviour
{
    public Canvas introCanvas;
    public void ContineButtonPressed()
    {
        introCanvas.gameObject.SetActive(false);
    }
}
