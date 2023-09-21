using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

/*
 *  This class controls your player. It deals with user inputs such as mouse and keyboard.
 *  It also controls the game state.
 *  
 */
public class Game : MonoBehaviour
{
    public GameObject player;
    public Room[] rooms;
    public AudioClip[] numbers;
    public Texture2D[] digits;
    public Camera playerCamera;
    public Texture2D circleCursor;

    public enum MODE { WALK, CONTROL};
    public MODE mode = MODE.WALK;

    Room currentRoom;
    CharacterController controller;
    float speed = 4; //walking speed
    float gravity = 0; //current falling speed due to gravity

    public static Game instance;

    // Start is called before the first frame update
    void Start()
    {
        instance = this;
        controller = player.GetComponent<CharacterController>();
        Cursor.lockState = CursorLockMode.Locked;
        for (int i = 0; i < rooms.Length; i++)
        {
            rooms[i].ResetCode();
            var panel = rooms[i].GetComponent<Panel>();
            if ( panel != null) panel.callback = GotNumber;
        }
        currentRoom = rooms[0];
    }


    //This is called from the panel once a digit has been entered. It gives the predicted number and the probability:
    void GotNumber(Room room, int n, float probability)
    {
        GetComponent<AudioSource>().PlayOneShot(numbers[n]);
        Debug.Log("Predicted number " + n + "\nProbability " + (probability * 100) + "%");

        //now we need to check if this code is correct:
        (bool correct, bool completed) = room.CheckCode(n);
        if (!correct)
        {
            //The guess is not correct so sound the alarm:
            currentRoom = room;
            Invoke("SoundAlarm", 0.5f);
        }
        if (completed)
        {
            if (room.doorState == Room.DOOR_STATE.CLOSED)
            {
                currentRoom = room;
                Invoke("PlayMessage", 1f); 
            }
            room.OpenDoor();
        }
    }

    //Sound the alarm and reset the code to something else:
    void SoundAlarm()
    {
        currentRoom.GetComponent<Panel>().SoundAlarm();
        currentRoom.ResetCode();
    }

    void PlayMessage()
    {
        if (currentRoom.message != null)
        {
            GetComponent<AudioSource>().PlayOneShot(currentRoom.message);
        }
    }

    void Update()
    {
        // Player movement:
        float mouseSensitivity = 1f;
        float vert = Input.GetAxis("Vertical");
        float horiz = Input.GetAxis("Horizontal");
        float mouseX = Input.GetAxis("Mouse X") * mouseSensitivity;

        if (Input.GetKeyDown(KeyCode.C) && Input.GetKey(KeyCode.LeftControl)) // cheat mode open all doors! Control+C
        {
            for (int i = 0; i < rooms.Length; i++)
                rooms[i].OpenDoor();
        }

        float factor = 1;
        if (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift)) factor = 2;

        gravity -= 9.81f * Time.deltaTime;
        controller.Move((player.transform.forward * vert + player.transform.right * horiz) * Time.deltaTime * speed * factor + player.transform.up * (gravity) * Time.deltaTime);
        if (controller.isGrounded) gravity = 0;
   
        if (mode == MODE.WALK)
        {
            controller.transform.Rotate(Vector3.up, mouseX);
        }

        // This toggles between using the mouse to look and using the mouse to draw on the screen
        if (Input.GetKeyDown(KeyCode.Space))
        {
            switch (mode)
            {
                case MODE.WALK:
                    mode = MODE.CONTROL;
                    Cursor.lockState = CursorLockMode.None;
                    Cursor.SetCursor(circleCursor, new Vector2(16, 16), CursorMode.Auto);
                    break;
                case MODE.CONTROL:
                    mode = MODE.WALK;
                    Cursor.lockState = CursorLockMode.Locked;
                    break;
            }
        }

        // Press escape to exit the game:
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Cursor.lockState = CursorLockMode.None;
            Application.Quit();
        }
    }
}
