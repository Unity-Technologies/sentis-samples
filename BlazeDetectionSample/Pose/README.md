# BlazePose in Sentis

BlazePose is a fast, light-weight hand detector from Google Research. Pretrained models are available as part of Google's [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) framework.

![](../images/pose.jpg)

The BlazePose models have been converted from TFLite to ONNX for use in Sentis using [tf2onnx](https://github.com/onnx/tensorflow-onnx) with the default export parameters. Three variants of the landmarker model (lite, full, heavy) are provided which can be interchanged. The larger models may provide more accurate results but take longer to run.

## Functional API

The BlazePose detector model takes a (1, 224, 224, 3) input image tensor and outputs a (1, 2254, 12) boxes tensor and a (1, 2254, 1) scores tensor.

Each of the 2254 boxes consists of:
- [x position, y position, width, height] for the head bounding box. The position is relative to the anchor position for the given index, these are precalculated and loaded from a csv file.
- [x position, y position] for each of 4 body keypoints relative to the anchor position.

We adapt the model using the Sentis functional API to apply arg max to filter the box with the highest score.
```
var detectionScores = ScoreFiltering(rawScores, 100f); // (1, 2254, 1)
var bestScoreIndex = Functional.ArgMax(rawScores, 1).Squeeze();

var selectedBoxes = Functional.IndexSelect(rawBoxes, 1, bestScoreIndex).Unsqueeze(0); // (1, 1, 12)
var selectedScores = Functional.IndexSelect(detectionScores, 1, bestScoreIndex).Unsqueeze(0); // (1, 1, 1)
```

The BlazePose landmarker model takes a (1, 256, 256, 3) input image tensor cropped to the detected body and outputs a (1, 165) tensor consisting of the x, y, and z coordinates and visibility and presence for each of 33 pose keypoints. We use this model without adaptation.

## Model inference

We use the dimensions of the texture to set up an affine transformation matrix to go from the 224x224 tensor coordinates to the image coordinates. We then fill the input tensor using a compute shader with this affine transformation, points outside the image will correspond to zeros in the input tensor.
```
var size = Mathf.Max(texture.width, texture.height);

// The affine transformation matrix to go from tensor coordinates to image coordinates
var scale = size / (float)detectorInputSize;
var M = BlazeUtils.mul(BlazeUtils.TranslationMatrix(0.5f * (new Vector2(texture.width, texture.height) + new Vector2(-size, size))), BlazeUtils.ScaleMatrix(new Vector2(scale, -scale)));
BlazeUtils.SampleImageAffine(texture, m_DetectorInput, M);

m_PoseDetectorWorker.Schedule(m_DetectorInput);
```

Execution is scheduled using an [Awaitable](https://docs.unity3d.com/6000.0/Documentation/ScriptReference/Awaitable.html) and the output tensors are downloaded and awaited. This frees up the main thread while the GPU computation and download takes place.
```
var outputIdxAwaitable = (m_PoseDetectorWorker.PeekOutput(0) as Tensor<int>).ReadbackAndCloneAsync();
var outputScoreAwaitable = (m_PoseDetectorWorker.PeekOutput(1) as Tensor<float>).ReadbackAndCloneAsync();
var outputBoxAwaitable = (m_PoseDetectorWorker.PeekOutput(2) as Tensor<float>).ReadbackAndCloneAsync();

using var outputIdx = await outputIdxAwaitable;
using var outputScore = await outputScoreAwaitable;
using var outputBox = await outputBoxAwaitable;
```
The output tensors of the detector model are now on the CPU and can be read. If the score passes our score threshold, we use the keypoint positions to set up a second affine transformation. This is calculated so that the body will be centred, rotated and scaled to fill the landmarker input tensor. We use the box and keypoint positions to set the transforms on bounding box and circle for visualization.

![](../images/pose_landmarker_input.png)
```
var face_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBox[0, 0, 0], outputBox[0, 0, 1]));
var faceTopRight_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBox[0, 0, 0] + 0.5f * outputBox[0, 0, 2], outputBox[0, 0, 1] + 0.5f * outputBox[0, 0, 3]));

var kp1_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBox[0, 0, 4 + 2 * 0 + 0], outputBox[0, 0, 4 + 2 * 0 + 1]));
var kp2_ImageSpace = BlazeUtils.mul(M, anchorPosition + new float2(outputBox[0, 0, 4 + 2 * 1 + 0], outputBox[0, 0, 4 + 2 * 1 + 1]));
var delta_ImageSpace = kp2_ImageSpace - kp1_ImageSpace;
var dscale = 1.25f;
var radius = dscale * math.length(delta_ImageSpace);
var theta = math.atan2(delta_ImageSpace.y, delta_ImageSpace.x);
var origin2 = new float2(0.5f * landmarkerInputSize, 0.5f * landmarkerInputSize);
var scale2 = radius / (0.5f * landmarkerInputSize);
var M2 = BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.mul(BlazeUtils.TranslationMatrix(kp1_ImageSpace), BlazeUtils.ScaleMatrix(new float2(scale2, -scale2))), BlazeUtils.RotationMatrix(0.5f * Mathf.PI - theta)), BlazeUtils.TranslationMatrix(-origin2));
BlazeUtils.SampleImageAffine(texture, m_LandmarkerInput, M2);

var boxSize = 2f * (faceTopRight_ImageSpace - face_ImageSpace);

posePreview.SetBoundingBox(true, ImageToWorld(face_ImageSpace), boxSize / m_TextureHeight);
posePreview.SetBoundingCircle(true, ImageToWorld(kp1_ImageSpace), radius / m_TextureHeight);

m_PoseLandmarkerWorker.Schedule(m_LandmarkerInput);
```
The output tensor of the landmarker model is asynchronously downloaded and once the values are on the CPU we use them together with the affine transformation matrix to set the transforms on the keypoints for visualization.

## WebGPU
Unity 6 supports access to the WebGPU backend in early access. Sentis has full support for running models on the web using the WebGPU backend. Discover how to gain early access and test WebGPU in our [graphics forum](https://discussions.unity.com/t/early-access-to-the-new-webgpu-backend-in-unity-2023-3/933493).

![](../images/pose_webgpu.png)