using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

/*
 * Handles game logic in regards of the room. 
 *  - opening and closing of the doors.
 *  - setting the various clues in each room.
 *  - also deal with the audio messages.
 *  - lights above the panel showing the correct answers.
 */
public class Room : MonoBehaviour
{
    public GameObject door,door2;
    public GameObject panel;
    public GameObject[] lights;
    public GameObject[] clue;
    public AudioClip message;

    public bool fruitCodes = false; // specific code for fruit level
    public int[] code; // secret code
    public int codePosition = 0; //e.g. is it the first, second or third digit you are entering
    public enum DOOR_STATE { OPEN, CLOSED, OPENING, CLOSING };
    public DOOR_STATE doorState = DOOR_STATE.CLOSED;

    // supported digits
    int[] numbers = new[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    int[] fruitNumbers = new int[] { 1, 2, 3, 4, 5, 6 };

    // door state (position/opening time/ width)
    float startTimeOfDoorState = 0;
    Vector3 doorClosedPosition;
    Vector3 door2ClosedPosition;
    float timeOfDoorOpening = 0.5f;
    float doorWidth = 4f;

    void Start()
    {
        if (door) doorClosedPosition = door.transform.position;
        if (door2) door2ClosedPosition = door2.transform.position;
    }

    // Reset code randomly and set the clue images in the room
    public void ResetCode()
    {
        code = new int[] { 0, 0, 0 };

        int[] randomNumbers = GetRandomizedNumbers();

        // Assign the first three random numbers in our randomized array to the three codes and set the clue materials
        for (int i = 0; i < code.Length; i++)
        {
            code[i] = randomNumbers[i];

            if (i >= clue.Length)
                return;
            
            var clueMaterial = clue[i].GetComponent<Renderer>().material;
            if (fruitCodes)
            {
                // changes the image of the fruit clue
                clueMaterial.mainTextureOffset = new Vector2((code[i] - 1) / 6f, 0);
            }
            else
            {
                clueMaterial.mainTexture = Game.instance.digits[code[i]];
            }
        }
    }

    // We want three different numbers. This can be done by shuffling the numbers 0..9 then picking the first three.
    // Or in the case of the fruit we shuffle the number 1..6
    int[] GetRandomizedNumbers()
    {
        if (fruitCodes)
        {
            return fruitNumbers.OrderBy((x) => Random.Range(0, 1f)).ToArray();
        }
        else
        {
            return numbers.OrderBy((x) => Random.Range(0, 1f)).ToArray();
        }
    }

    public void OpenDoor()
    {
        if (!door) return;
        if (doorState == DOOR_STATE.CLOSED)
        {          
            doorState = DOOR_STATE.OPENING;
            startTimeOfDoorState = Time.time;
            door.GetComponent<AudioSource>().Play();
        }
        if (doorState == DOOR_STATE.OPEN)
        {
            doorState = DOOR_STATE.CLOSING;
            startTimeOfDoorState = Time.time;
            door.GetComponent<AudioSource>().Play();
        }
    }

    // Check if the code is correct and if we have completed the full code
    public (bool correct, bool completed) CheckCode(int digitGuess)
    {
        bool isCorrectGuess = (digitGuess == code[codePosition]);

        if (isCorrectGuess) 
        {
            // turn lights green
            for (int i = 0; i < lights.Length; i++)
            {
                var lightMaterial = lights[i].GetComponent<Renderer>().material;
                lightMaterial.SetColor("_EmissionColor", i <= codePosition ? Color.green : Color.black);
            }

            codePosition++;
            bool everyDigitCorrect = (codePosition == code.Length); // every digit correct
            if (everyDigitCorrect) 
            {
                codePosition = 0;
                return (correct:true, completed:true);
            }
            else
            {
                return (correct: true, completed:false);
            }
        }
        else //wrong guess
        {
            // turn off lights
            for (int i = 0; i < lights.Length; i++)
            {
                var lightMaterial = lights[i].GetComponent<Renderer>().material;
                lightMaterial.SetColor("_EmissionColor", Color.black);
            }
            codePosition = 0;
            return (correct: false, completed: false);
        }
    }


    void Update()
    {
        // animate doors opening and closing
        float t = (Time.time - startTimeOfDoorState) / timeOfDoorOpening; 
        if (doorState == DOOR_STATE.OPENING)
        {
            OpeningDoor(t);
        }
        if (doorState == DOOR_STATE.CLOSING)
        {
            ClosingDoor(t);
        }
    }

    void OpeningDoor(float t)
    {
        if (!door && !door2) return;

        door.transform.position = doorClosedPosition + door.transform.right * Mathf.Min(1, t) * doorWidth;
        door2.transform.position = door2ClosedPosition - door2.transform.right * Mathf.Min(1, t) * doorWidth;

        //is door fully open?
        if (t > 1)
        {
            doorState = DOOR_STATE.OPEN;
        }
    }

    void ClosingDoor(float t)
    {
        if (!door || !door2) return;

        door.transform.position = doorClosedPosition + door.transform.right * Mathf.Max(0, 1 - t) * doorWidth;
        door2.transform.position = doorClosedPosition - door2.transform.right * Mathf.Max(0, 1 - t) * doorWidth;

        //is door fully closed?
        if (t > 1)
        {
            doorState = DOOR_STATE.CLOSED;
        }
    }
}
