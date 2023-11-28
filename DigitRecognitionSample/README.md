# Escape Room Sample
 
Escape room adventure showing how to integrate digit recognition into the gameplay loop.


![image info](./Documentation/main.png)

## Gameplay Idea

We want a game where the player is stuck in a room and to escape they need to draw a code on a codepad.

![image info](./Documentation/gameplay.png)

## Runtime Inference

To solve this problem we leverage a small Neural Network.

It takes as input the code-pad texture.

After running inference we get the most likely digit.

We use this to feedback into the gameplay loop

![image info](./Documentation/runtime-inference.png)


##  Video Tutorial

[![IMAGE ALT TEXT HERE](../Documentation/video-image.png)](https://www.youtube.com/watch?v=IofX0CAYdmU)
