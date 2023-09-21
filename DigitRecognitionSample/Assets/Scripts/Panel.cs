using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using UnityEngine.UI;
using System.Linq;

/*
 *  Manages Pannel
 *    - drawing things to the screen.
 *    - alarm animations.
 */
public class Panel : MonoBehaviour
{
    
    public GameObject screen;
    public Text predictionText, probabilityText;
    public MNISTEngine mnist;
    public GameObject alarm;
    public Light mainLight;
    public System.Action<Room, int, float> callback;


    // code pad where the digit is drawn onto
    Texture2D drawableTexture;
    const int imageWidth = 28; //width and height of input image
    float[] imageData = new float[imageWidth * imageWidth];
    byte[] zeroes = new byte[imageWidth * imageWidth * 3]; // blank screen
    Vector3 lastCoord; // last position of mouse on screen

    // digit recognition
    int predictedNumber;
    float probability;

    float timeOfLastEntry = float.MaxValue;
    float clearTime = 0.5f; // time digit is on screen before it is cleared
    Camera lookCamera;

    // alarm state
    enum STATE { NORMAL, ALARM };
    STATE state = STATE.NORMAL;
    float startTimeOfState = 0;
    float alarmPeriod = 2f; // number of seconds for alarm
    Color originalLightColor;

    // room state
    Room room;


    void Start()
    {
        lookCamera = Camera.main;

        // code pad texture which will be drawn into:
        drawableTexture = new Texture2D(imageWidth, imageWidth, TextureFormat.RGB24, false);
        drawableTexture.wrapMode = TextureWrapMode.Clamp;
        drawableTexture.filterMode = FilterMode.Point;

        ClearTexture();

        // emission map for glowing digits
        screen.GetComponent<Renderer>().material.SetTexture("_EmissionMap", drawableTexture);

        room = GetComponent<Room>();
        originalLightColor = mainLight.color;

        predictionText.text = "?";
    }

    public void SoundAlarm()
    {
        state = STATE.ALARM;
        alarm.GetComponent<AudioSource>().Play();
        startTimeOfState = Time.time;
    }

    void ClearTexture()
    {
        drawableTexture.LoadRawTextureData(zeroes);
        drawableTexture.Apply();
    }

    // Calls the neural network to get the probabilities of different digits then selects the most likely
    void Infer()
    {
        var probabilityAndIndex = mnist.GetMostLikelyDigitProbability(drawableTexture);

        probability = probabilityAndIndex.Item1;
        predictedNumber = probabilityAndIndex.Item2;
        predictionText.text = predictedNumber.ToString();
        if (probabilityText) probabilityText.text = Mathf.Floor(probability * 100) + "%";
    }

    // Draws a line on the panel by simply drawing a sequence of pixels
    void DrawLine(Vector3 startp, Vector3 endp)
    {
        int steps = (int)((endp - startp).magnitude * 2 + 1); 
        for(float a = 0; a <= steps; a++)
        {
            float t = a * 1f / steps;
            DrawPoint(startp * (1 - t) + endp * t , 2, Color.white);
        }
    }

    // Draws either a single pixel or a 2x2 pixel for a thicker line
    void DrawPoint(Vector3 coord, int thickness, Color color)
    {
        //clamp the values so it doesn't touch the border
        float x = Mathf.Clamp(coord.x, thickness, imageWidth - thickness);
        float y = Mathf.Clamp(coord.y, thickness, imageWidth - thickness);

        switch (thickness)
        {
            case 1:
                DrawPixel((int)x, (int)y, color);
                break;
            case 2:
            default:
                int x0 = Mathf.Max(0, (int)(x - 0.5f));
                int x1 = Mathf.Min(imageWidth - 1, (int)(x + 0.5f));
                int y0 = Mathf.Max(0, (int)(y - 0.5f));
                int y1 = Mathf.Min(imageWidth - 1, (int)(y + 0.5f));
                DrawPixel(x0, y0, color);
                DrawPixel(x1, y0, color);
                DrawPixel(x0, y1, color);
                DrawPixel(x1, y1, color);
                break;
        }
    }

    void DrawPixel(int x,int y,Color color)
    {
        drawableTexture.SetPixel(x, y, color);
    }

    public void ScreenMouseDown(RaycastHit hit)
    {
        if (Game.instance && Game.instance.mode != Game.MODE.CONTROL) return;
        Vector2 uv = hit.textureCoord;
        Vector3 coords = uv * imageWidth;
        lastCoord = coords;
        timeOfLastEntry = Time.time;
    }

    public void ScreenGetMouse(RaycastHit hit)
    {
        if (Game.instance &&  Game.instance.mode != Game.MODE.CONTROL) return;
        Vector2 uv = hit.textureCoord;
        Vector3 coords = uv * imageWidth;

        DrawLine(lastCoord, coords);
        lastCoord = coords;
        drawableTexture.Apply();

        timeOfLastEntry = Time.time;
        // Run the inference every frame since it is very fast
        Infer();
    }

    void Update()
    {
        if (state == STATE.ALARM)
        {
            float t = Time.time - startTimeOfState;
            if (t < alarmPeriod)
            {
                AnimateAlarm(t);
            }
            else
            {
                StopAlarm();
            }
        }

        // After a certain time we want to clear the panel:
        if ((Time.time - timeOfLastEntry) > clearTime)
        {
            if (callback != null) callback(room, predictedNumber, probability);
            ClearTexture();
            timeOfLastEntry = float.MaxValue;
        }

    }

    void AnimateAlarm(float t)
    {
        Color lightColor = new Color(Mathf.Pow(Mathf.Cos(t * Mathf.PI * 4), 2), 0, 0);
        for (int i = 0; i < room.lights.Length; i++)
        {
            room.lights[i].GetComponent<Renderer>().material.SetColor("_EmissionColor", lightColor);
        }
        mainLight.color = lightColor;
    }

    void StopAlarm()
    {
        alarm.GetComponent<AudioSource>().Stop();
        state = STATE.NORMAL;
        mainLight.color = originalLightColor;
        for (int i = 0; i < room.lights.Length; i++)
        {
            room.lights[i].GetComponent<Renderer>().material.SetColor("_EmissionColor", Color.black);
        }
        predictionText.text = "?";
    }
}
