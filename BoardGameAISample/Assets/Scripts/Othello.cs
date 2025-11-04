using UnityEngine;
using Unity.InferenceEngine;
using UnityEngine.UI;


// Main board game logic handling
// Each turn either the player move or the neural network guess the best possible move for the opponent to play.
// Probabilities for the next best move of the opponent are displayed graphically.
// Output of the model which gives estimates who is winning given the current state of the game is used to trigger quotes for the characters.
public class Othello : MonoBehaviour
{
    public ModelAsset model;
    Worker m_Engine;

    // Board logic
    public AudioClip pieceDown, illegalMoveBuzzer;
    public Text subtitleText;
    public GameObject subtitleBackground;
    public GameObject piece;
    public Material[] materials;
    public GameObject probabilityBox , probabilityHolder;
    public OpponentAvatar[] opponentAvatars;
    public AudioSource fanfare;
    enum GameMode { CHOOSE_CHARACTER, PLAY_GAME};
    GameMode m_GameMode = GameMode.CHOOSE_CHARACTER;

    // Board orientation + animation
    Vector3 m_DirectionX = Vector3.right;
    Vector3 m_DirectionY = Vector3.forward;
    Vector3 m_DirectionZ = Vector3.up;
    float m_zOffset = 0f;

    float m_PauseTime = 1f; // time to wait as spirit makes their move
    float m_SimulationSpeed = 0.25f;

    float m_TimeOfSubtitle = 0;
    float m_SubtitleLength = 2f;

    // Camera movement
    enum CameraMove { TO_BOARD, TO_OPPONENTS, NONE };
    CameraMove cameraMove = CameraMove.NONE;
    float cameraAngleUp = 0;
    float cameraAngleLR = 0;

    // AI difficulty
    // A temperature of 0 = completely random
    // A high temperature e.g. 2 = chooses most probable move most of the time
    public float m_AIDifficultyTemperature = 1f;

    // Board state
    const int kBoardDimension = 8; // The size of the board 8x8
    const int kBLACK = 1; // Human
    const int kRED = -1;  // Spirit
    int m_CurrentTurn = kBLACK; // keep track of who's turn it is to play

    Tensor<float> m_Data;
    Tensor<float> m_LegalMoves;
    Tensor<float> m_MoveProbabilities = null;
    GameObject[] m_Pieces = new GameObject[kBoardDimension * kBoardDimension];

    int m_PassesInARow = 0;
    // the number of pieces on the board
    int m_RedPieces = 0;
    int m_BlackPieces = 0;

    // stores the person who was winning last move (for use with phrases)
    int m_LastWinning = 0;

    void Start()
    {
        m_Data = new Tensor<float>(new TensorShape(1, 1, kBoardDimension, kBoardDimension));
        m_LegalMoves = new Tensor<float>(new TensorShape(kBoardDimension * kBoardDimension + 1));

        CreateBoard();
    }

    void NextMove()
    {
        CancelInvoke();

        GetLegalMoves();

        // Check if player or spirit has won
        if (m_RedPieces + m_BlackPieces == kBoardDimension * kBoardDimension || m_PassesInARow>1)
        {
            fanfare.Play();
            if (m_RedPieces > m_BlackPieces)
            {
                SetSubtitle("I win!");
            }
            else if (m_RedPieces < m_BlackPieces)
            {
                SetSubtitle("You win!");
            }
            else if (m_RedPieces == m_BlackPieces)
            {
                SetSubtitle("It's a draw.");
            }
            Invoke("ResetBoard", 2f);
        }
        else
        {
            if (m_CurrentTurn == kRED)
            {
                if(m_LegalMoves[kBoardDimension * kBoardDimension] == 1)
                {
                    SetSubtitle("I can't move so I pass this turn.");
                }
                Invoke("ComputerMove", m_PauseTime);
            }
            if (m_CurrentTurn == kBLACK && m_LegalMoves[kBoardDimension * kBoardDimension] == 1)
            {
                SetSubtitle("You can't move so you pass this turn.");
                //we do passing automatically
                ComputerMove();
            }
        }
    }

    Vector3 GetPiecePosition(int y, int x)
    {
        return (x - kBoardDimension / 2 + 0.5f) * m_DirectionX + (-y + kBoardDimension / 2 - 0.5f) * m_DirectionY + m_zOffset * m_DirectionZ;
    }

    void RotateCameraToBoard(float speed)
    {
        Camera.main.transform.rotation = Quaternion.RotateTowards(Camera.main.transform.rotation, Quaternion.Euler(42, 0, 0), Time.deltaTime* speed);
    }
    void RotateCameraUp(float speed)
    {
        Camera.main.transform.rotation = Quaternion.RotateTowards(Camera.main.transform.rotation, Quaternion.Euler(5, 0, 0), Time.deltaTime * speed);
    }

    void CreateBoard()
    {
        for(int y = 0; y < kBoardDimension; y++)
        {
            for(int x = 0; x < kBoardDimension; x++)
            {
                var newSquare = Instantiate(piece);
                newSquare.transform.SetParent(transform, false);
                newSquare.transform.localPosition = GetPiecePosition(y, x);
                m_Pieces[y * kBoardDimension + x] = newSquare;
            }
        }
    }

    public void LevelOptionSelected(int playerID, int level)
    {
        cameraMove = CameraMove.TO_BOARD;
        if (playerID == kRED)
        {
            opponentAvatars[0].mode = (0 == level ? OpponentAvatar.Mode.Forward : OpponentAvatar.Mode.Back);
            opponentAvatars[1].mode = (1 == level ? OpponentAvatar.Mode.Forward : OpponentAvatar.Mode.Back);
            opponentAvatars[2].mode = (2 == level ? OpponentAvatar.Mode.Forward : OpponentAvatar.Mode.Back);
        }

        Debug.Log("Level=" + level);
        if (level == 0) m_AIDifficultyTemperature = 0;
        if (level == 1) m_AIDifficultyTemperature = 0.5f;
        if (level == 2) m_AIDifficultyTemperature = 2f;

        CreateEngine();

        if (m_GameMode == GameMode.CHOOSE_CHARACTER)
        {
            m_GameMode = GameMode.PLAY_GAME;
            ResetBoard();
        }

        NextMove();
    }

    void CreateEngine()
    {
        m_Engine?.Dispose();

         // Load in the neural network that will make the move predictions for the spirit + create Sentis
        var othelloModel = ModelLoader.Load(model);

        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(othelloModel);
        var outputs = Functional.Forward(othelloModel, inputs);
        var bestMove = outputs[0];
        var boardState = outputs[1];

        // Ensure legal moves are considered when computing best move.
        var legal = graph.AddInput(DataType.Float, new TensorShape(kBoardDimension * kBoardDimension + 1));
        // Convert outputs to probabilities
        bestMove = Functional.Exp(bestMove * m_AIDifficultyTemperature);
        // Mask out illegal moves
        bestMove = (0.0001f + bestMove) * legal;
        // Normalize probabilities so they sum to 1
        var redSum = Functional.ReduceSum(bestMove, new int[] { 1 }, true);
        bestMove /= redSum;

        var bestMoveModel = graph.Compile(boardState, bestMove);

        m_Engine = new Worker(bestMoveModel, BackendType.CPU);
    }

    void ResetBoard()
    {
        CancelInvoke();
        for(int i = 0; i < m_Data.shape.length; i++)
            m_Data[i] = 0.0f;

        // Starting position:
        m_Data[(kBoardDimension / 2), (kBoardDimension / 2)] = kRED;
        m_Data[(kBoardDimension / 2 - 1), (kBoardDimension / 2)] = kBLACK;
        m_Data[(kBoardDimension / 2), (kBoardDimension / 2 - 1)] = kBLACK;
        m_Data[(kBoardDimension / 2 - 1), (kBoardDimension / 2 - 1)] = kRED;

        m_PassesInARow = 0;
        m_LastWinning = 0;

        SetColors(kRED);
        m_CurrentTurn = kBLACK;
        SetSubtitle("Let's play. You begin.");
        NextMove();
    }
    void SetColors(int turn)
    {
        for(int i = 0; i < kBoardDimension * kBoardDimension; i++)
        {
            m_Pieces[i].GetComponent<Renderer>().material = m_Data[i] == -turn ? materials[0] : m_Data[i] == turn ? materials[1] : materials[2];
        }
    }

    int SelectRandomMove()
    {
        float randnum = UnityEngine.Random.Range(0, 1f);
        float s = 0;
        for(int i = 0; i < m_MoveProbabilities.shape.length; i++)
        {
            s += m_MoveProbabilities[i];
            if (s > randnum) return i;
        }
        return m_MoveProbabilities.shape.length - 1;
    }

    void FlipBoard()
    {
        for (int i = 0; i < m_Data.shape.length; i++)
            m_Data[i] *= -1;
    }

    // The spirit makes a move using neural network
    void ComputerMove()
    {
        // The network is always form the point of view that the current player = 1 and opponent = -1
        FlipBoard();

        m_Engine.Schedule(m_Data, m_LegalMoves);

        // estimate who is winning:
        using var boardState = (m_Engine.PeekOutput(0) as Tensor<float>).ReadbackAndClone();
        // predict best move:
        m_MoveProbabilities?.Dispose();
        m_MoveProbabilities = (m_Engine.PeekOutput(1) as Tensor<float>).ReadbackAndClone();

        float boardValue = boardState[0, 0];
        bool blackIsWinning = -m_CurrentTurn * boardValue < 0;

        //convert the boardValue [-1,1] into a more human readable number:
        int percent = (int)(Mathf.Pow(Mathf.Abs(boardValue), 10f) * 50 + 50);

        DisplayPhrases(blackIsWinning, percent);

        DisplayProbabilities();

        Invoke("MakeMove", m_PauseTime);
    }


    void DisplayPhrases(bool blackIsWinning, int percent)
    {
        int winning = 0;

        if (percent >= 60)
        {
            if (blackIsWinning)
            {
                winning = kBLACK;
            }
            else
            {
                winning = kRED;
            }
        }
        if (m_LastWinning == kBLACK && winning == kRED)
        {
            SetSubtitle("The tables have turned! I am winning!");
        }
        if (m_LastWinning == 0 && winning == kRED)
        {
            SetSubtitle("Aha! I take the lead!");
        }
        if (m_LastWinning == 0 && winning == kBLACK)
        {
            SetSubtitle("You are doing well.");
        }
        if (m_LastWinning == kRED && winning == kBLACK)
        {
            SetSubtitle("Good move. You have me at a disadvantage.");
        }
        if(m_LastWinning != 0 && winning == 0)
        {
            SetSubtitle("It's going to be a close game.");
        }
        m_LastWinning = winning;
    }

    void SetSubtitle(string text)
    {
        subtitleBackground.SetActive(true);
        m_TimeOfSubtitle = Time.time;
        subtitleText.text = text;
    }

    void HideSubtitle()
    {
        subtitleBackground.SetActive(false);
        m_TimeOfSubtitle = -float.MaxValue;
        subtitleText.text = "";
    }

    void DisplayProbabilities()
    {
        ClearProbabilityDisplay();
        for (int i = 0; i < kBoardDimension * kBoardDimension; i++)
        {
            if (m_MoveProbabilities[i] > 0)
            {
                var newFlame = Instantiate(probabilityBox);
                newFlame.transform.SetParent(probabilityHolder.transform);
                newFlame.transform.localPosition = GetPiecePosition(i / kBoardDimension, i % kBoardDimension);
                newFlame.transform.localScale = new Vector3(1, m_MoveProbabilities[i] * 2, 1);

                //change colour of flames (green is more probable white is less probable)
                var particleSystem = newFlame.transform.GetChild(0).GetComponent<ParticleSystem>();
                var particleProperties = particleSystem.main;
                particleProperties.startColor = Color.Lerp(new Color(1f,1f,1f,0.1f), Color.green, m_MoveProbabilities[i] * 2);
                //re-simulate to flush out the old particle colours:
                particleSystem.Simulate(1f);
                particleSystem.Play();
            }
        }
    }

    void ClearProbabilityDisplay()
    {
        foreach (Transform t in probabilityHolder.transform)
        {
            Destroy(t.gameObject);
        }
    }

    void MakeMove()
    {
        ClearProbabilityDisplay();
        int moveIndex = SelectRandomMove();

        if (m_LegalMoves[moveIndex] == 0)
        {
            // should never happen
            Debug.Log("*****WARNING ILLEGAL MOVE CHOSEN!!!*****");
            return;
        }

        if (moveIndex >= kBoardDimension * kBoardDimension)
        {
            Debug.Log("************PASS*************");
            m_PassesInARow++;
        }
        else
        {
            GetComponent<AudioSource>().PlayOneShot(pieceDown);
            m_PassesInARow = 0;
            m_Data[moveIndex] = 1;
            m_Pieces[moveIndex].GetComponent<Piece>().BeginAnimation();
            FlipColors(moveIndex / kBoardDimension, moveIndex % kBoardDimension, 1);
        }

        SetColors(m_CurrentTurn);
        m_CurrentTurn = -m_CurrentTurn;
        NextMove();
    }

    void GetLegalMoves()
    {
        m_RedPieces = 0;
        m_BlackPieces = 0;
        bool moveAvailable = false;
        for(int y = 0; y < kBoardDimension; y++)
        {
            for(int x = 0; x < kBoardDimension; x++)
            {
                m_RedPieces += m_Data[y, x] == m_CurrentTurn ? 1 : 0;
                m_BlackPieces += m_Data[y, x] == -m_CurrentTurn ? 1 : 0;
                bool legal = m_Data[y, x] == 0 && FlipColors(y, x, -1, true) == 1;
                m_LegalMoves[y * kBoardDimension + x] = legal ? 1 : 0;
                if (legal) moveAvailable = true;
            }
        }
        m_LegalMoves[kBoardDimension * kBoardDimension] = moveAvailable ? 0 : 1; //can pass ONLY if no moves available
    }

    int FlipColors(int y, int x, int turn, bool checkonly=false)
    {
        for(int dx = -1; dx <= 1; dx++)
        {
            for(int dy = -1; dy <= 1; dy++)
            {
                if (dx == 0 && dy == 0) continue;
                // check for possible line
                (int X, int Y) = (x + dx, y + dy);
                int enemyPieces = 0;
                // check for a line of enemy pieces in direction (dx,dy):
                while (Y >= 0 && X >= 0 && X < kBoardDimension && Y < kBoardDimension && m_Data[Y * kBoardDimension + X] == -turn)
                {
                    X += dx; Y += dy;
                    enemyPieces++;
                }
                // if we found a line of enemy pieces capped with one of your pieces we can flip them to your color:
                if (Y >= 0 && X >= 0 && X < kBoardDimension && Y < kBoardDimension && m_Data[Y * kBoardDimension + X] == turn && enemyPieces > 0)
                {
                    if (checkonly) return 1;
                    X = x + dx; Y = y + dy;
                    while (m_Data[Y * kBoardDimension + X] == -turn)
                    {
                        m_Data[Y * kBoardDimension + X] = turn;
                        if (!checkonly)
                        {
                            m_Pieces[Y * kBoardDimension + X].GetComponent<Piece>().BeginFlipAnimation();
                        }
                        X += dx; Y += dy;
                    }
                }
            }
        }
        return 0;
    }

    // move the camera back up so you can choose a different opponent:
    public void OnUpButtonClicked()
    {
        cameraMove = CameraMove.TO_OPPONENTS;
    }

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }

        if (cameraMove == CameraMove.TO_BOARD) RotateCameraToBoard(20f);
        if (cameraMove == CameraMove.TO_OPPONENTS) RotateCameraUp(20f);

        if (Time.time - m_TimeOfSubtitle > m_SubtitleLength)
        {
            HideSubtitle();
        }

        if (Input.GetKeyDown(KeyCode.Space))
        {
            //Let the computer take the move for us. If there is no move, pass.
            ComputerMove();
        }

        if (Input.GetMouseButtonDown(0))
        {
            MouseClicked();
        }

        if (m_GameMode == GameMode.CHOOSE_CHARACTER) return;

        if (Input.GetKeyDown(KeyCode.R))
        {
            ResetBoard();
        }

        if (Input.GetKeyDown(KeyCode.LeftShift))
        {
            cameraMove = CameraMove.NONE;
            Cursor.lockState = CursorLockMode.Locked;
            cameraAngleUp = Camera.main.transform.localEulerAngles.x;
            cameraAngleLR = Camera.main.transform.localEulerAngles.y;
        }
        else if (Input.GetKey(KeyCode.LeftShift))
        {
            float mouseSensititvy = 1f;
            float mouseX = Input.GetAxis("Mouse X") * mouseSensititvy;
            float mouseY = Input.GetAxis("Mouse Y") * mouseSensititvy;

            cameraAngleLR = cameraAngleLR + mouseX;

            cameraAngleUp = Mathf.Clamp(cameraAngleUp - mouseY, -45, 45);
            Camera.main.transform.localEulerAngles = new Vector3(cameraAngleUp, cameraAngleLR, 0);
        }
        if (Input.GetKeyUp(KeyCode.LeftShift))
        {
            Cursor.lockState = CursorLockMode.None;
        }

        if (Input.GetKeyDown(KeyCode.LeftArrow))
        {
            m_SimulationSpeed *= 2;
        }
        if (Input.GetKeyDown(KeyCode.RightArrow))
        {
            m_SimulationSpeed /= 2;
        }
    }

    bool HasParent(Transform t, string name)
    {
        if (t.name == name) return true;
        if (t.parent) return HasParent(t.parent, name);
        return false;
    }

    void MouseClicked()
    {
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        if (!Physics.Raycast(ray, out RaycastHit hit, 1000))
            return;

        GameObject go = hit.collider.gameObject;

        if (HasParent(go.transform, "Opponent Easy"))
        {
            LevelOptionSelected(kRED, 0);
        }
        if (HasParent(go.transform, "Opponent Medium"))
        {
            LevelOptionSelected(kRED, 1);
        }
        if (HasParent(go.transform, "Opponent Best"))
        {
            LevelOptionSelected(kRED, 2);
        }

        int index = System.Array.IndexOf(m_Pieces, go);
        if (index < 0)
            return;

        if (m_LegalMoves[index] == 1)
        {
            FlipBoard();
            GetComponent<AudioSource>().PlayOneShot(pieceDown);
            m_Data[index] = 1;
            m_Pieces[index].GetComponent<Piece>().BeginAnimation();
            FlipColors(index / kBoardDimension, index % kBoardDimension, 1);
            SetColors(m_CurrentTurn);
            m_CurrentTurn = -m_CurrentTurn;
            NextMove();
        }
        else
        {
            GetComponent<AudioSource>().PlayOneShot(illegalMoveBuzzer);
            Debug.Log("Can't go there");
        }
    }

    private void OnApplicationQuit()
    {
        CancelInvoke();
        m_Engine?.Dispose();
        m_MoveProbabilities?.Dispose();
        m_Data.Dispose();
        m_LegalMoves.Dispose();
    }
}
