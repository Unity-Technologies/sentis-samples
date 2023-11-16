using System;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Options for the formatting of the box data for `NonMaxSuppression`.
    /// </summary>
    public enum CenterPointBox
    {
        /// <summary>
        /// Use TensorFlow box formatting. Box data is [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are the normalized coordinates of any diagonal pair of box corners.
        /// </summary>
        Corners,
        /// <summary>
        /// Use PyTorch box formatting. Box data is [x_center, y_center, width, height].
        /// </summary>
        Center
    }

    /// <summary>
    /// Represents a `NonMaxSuppression` object detection layer. This calculates an output tensor of selected indices of boxes from input `boxes` and `scores` tensors, and bases the indices on the scores and amount of intersection with previously selected boxes.
    /// </summary>
    [Serializable]
    [Optimization.CPUFallback.CPUReadInputs(2, 3, 4)]
    public class NonMaxSuppression : Layer
    {
        /// <summary>
        /// The format in which the box data is stored in the `boxes` tensor as a `CenterPointBox`.
        /// </summary>
        public CenterPointBox centerPointBox;

        /// <summary>
        /// Initializes and returns an instance of `NonMaxSuppression` object detection layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="boxes">The name to use for the boxes tensor of the layer.</param>
        /// <param name="scores">The name to use for the scores tensor of the layer.</param>
        /// <param name="maxOutputBoxesPerClass">The name to use for an optional scalar tensor, with the maximum number of boxes to return for each class.</param>
        /// <param name="iouThreshold">The name to use for optional scalar tensor, with the threshold above which the intersect-over-union rejects a box.</param>
        /// <param name="scoreThreshold">The name to use for an optional scalar tensor, with the threshold below which the box score filters a box from the output.</param>
        /// <param name="centerPointBox">The format the `boxes` tensor uses to store the box data as a `CenterPointBox`. The default value is `CenterPointBox.Corners`.</param>
        public NonMaxSuppression(string name, string boxes, string scores, string maxOutputBoxesPerClass = null, string iouThreshold = null, string scoreThreshold = null, CenterPointBox centerPointBox = CenterPointBox.Corners)
        {
            this.name = name;
            if (scoreThreshold != null)
                this.inputs = new[] { boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold };
            else if (iouThreshold != null)
                this.inputs = new[] { boxes, scores, maxOutputBoxesPerClass, iouThreshold };
            else if (maxOutputBoxesPerClass != null)
                this.inputs = new[] { boxes, scores, maxOutputBoxesPerClass };
            else
                this.inputs = new[] { boxes, scores };
            this.centerPointBox = centerPointBox;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shape = new SymbolicTensorShape(SymbolicTensorDim.Unknown, new SymbolicTensorDim(3));
            return new PartialTensor(DataType.Int, shape);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var maxOutputBoxesPerClass = inputTensors.Length > 2 && inputTensors[2] != null ? inputTensors[2].ToReadOnlySpan<int>()[0] : 0;
            var iouThreshold = inputTensors.Length > 3 && inputTensors[3] != null ? inputTensors[3].ToReadOnlySpan<float>()[0] : 0f;
            var scoreThreshold = inputTensors.Length > 4 && inputTensors[4] != null ? inputTensors[4].ToReadOnlySpan<float>()[0] : 0f;
            var boxes = inputTensors[0] as TensorFloat;
            var scores = inputTensors[1] as TensorFloat;

            // Filter out boxes that have high intersection-over-union (IOU) overlap with previously selected boxes.
            // Bounding boxes with score less than scoreThreshold are removed.
            // maxOutputBoxesPerClass represents the maximum number of boxes to be selected per batch per class.
            // This algorithm is agnostic to where the origin is in the coordinate system and more generally is invariant to orthogonal transformations and translations of the coordinate system; thus translating or reflections of the coordinate system result in the same boxes being selected by the algorithm.
            // Bounding box format is indicated by attribute centerPointBox. Corners - diagonal y,x pairs (coords or normalized values). Center - center coords + width and height.
            // iouThreshold represents the threshold for deciding whether boxes overlap too much with respect to IOU. 0-1 range. 0 - no filtering.
            // The output is a set of integers indexing into the input collection of bounding boxes representing the selected boxes sorted in descending order and grouped by batch and class.

            ShapeInference.NonMaxSuppression(boxes.shape, scores.shape, iouThreshold);
            if (boxes.shape.HasZeroDims() || scores.shape.HasZeroDims() || maxOutputBoxesPerClass <= 0)
                return ctx.backend.NewOutputTensorInt(new TensorShape(0, 3));

            ArrayTensorData.Pin(boxes);
            ArrayTensorData.Pin(scores);

            // allocate the maximum possible output size tensor
            var outputData = new int[scores.shape[0] * scores.shape[1] * maxOutputBoxesPerClass * 3];
            // array of the current selected output indexes
            var selectedIndexes = new int[maxOutputBoxesPerClass];
            // array of the current selected output scores
            var selectedScores = new float[maxOutputBoxesPerClass];
            // keep a track of total output boxes
            int numberOfBoxes = 0;

            // find boxes to keep and then combine them into the single output tensor grouped by current batch and class
            for (int batch = 0; batch < scores.shape[0]; batch++)
            {
                for (int classID = 0; classID < scores.shape[1]; classID++)
                {
                    //keep a track of selected boxes per batch and class
                    int selectedBoxes = 0;
                    Array.Clear(selectedIndexes, 0, maxOutputBoxesPerClass);
                    Array.Clear(selectedScores, 0, maxOutputBoxesPerClass);
                    // iterate over input boxes for the current batch and class
                    for (int i = 0; i < scores.shape[2]; i++)
                    {
                        // check if the score is lower that the scoreThreshold
                        if (scores[batch, classID, i] < scoreThreshold)
                            continue;

                        // initialize insert index to last position
                        int insertIndex = selectedBoxes;
                        bool isIgnoreBox = false;

                        // compare input boxes to the already selected boxes
                        for (int j = 0; j < selectedBoxes; j++)
                        {
                            // if insert index is still default, i.e. box score is lower than previous sorted boxes, compare to see if this is the correct insert index
                            if ((insertIndex == selectedBoxes) && scores[batch, classID, i] > scores[batch, classID, selectedIndexes[j]])
                                insertIndex = j;

                            // if not excessive overlap with this box consider next box
                            if (NotIntersectOverUnion(
                                    boxes[batch, i, 0],
                                    boxes[batch, i, 1],
                                    boxes[batch, i, 2],
                                    boxes[batch, i, 3],
                                    boxes[batch, selectedIndexes[j], 0],
                                    boxes[batch, selectedIndexes[j], 1],
                                    boxes[batch, selectedIndexes[j], 2],
                                    boxes[batch, selectedIndexes[j], 3],
                                    centerPointBox, iouThreshold))
                                continue;

                            // new box has lower score than overlap box so do not output new box
                            if (insertIndex >= selectedBoxes)
                            {
                                isIgnoreBox = true;
                                break;
                            }

                            // new box has higher score than overlap box so remove overlap box from list, no need to shift memory if it is in final position
                            if (j < (maxOutputBoxesPerClass - 1))
                            {
                                // remove the overlaping box index and score values from the current selected box array by shifting the memory
                                // selectedIndexes/selectedScores = [x x x j y y y]
                                // <- shift y y y by one
                                // [x x x y y y]
                                unsafe
                                {
                                    fixed (int* dst = &selectedIndexes[j])
                                        UnsafeUtility.MemMove(dst, dst + 1, (maxOutputBoxesPerClass - (j + 1)) * sizeof(int));
                                    fixed (float* dst = &selectedScores[j])
                                        UnsafeUtility.MemMove(dst, dst + 1, (maxOutputBoxesPerClass - (j + 1)) * sizeof(int));
                                }
                            }
                            selectedBoxes--;
                            j--;
                        }

                        // either new box has lower score than an overlap box or there are already maxOutputBoxesPerClass with a better score, do not output new box
                        if (isIgnoreBox || insertIndex >= maxOutputBoxesPerClass)
                            continue;

                        // shift subsequent boxes forward by one in sorted array to make space for new box, no need if new box is after all boxes or or at end of array
                        if (insertIndex < selectedBoxes && insertIndex < (maxOutputBoxesPerClass - 1))
                        {
                            // shift memory to free a slot for a new box index and score values
                            // selectedIndexes/selectedScores = [x x x y y y]
                            // -> shift y y y by one
                            // [x x x insertIndex y y y]
                            unsafe
                            {
                                fixed (int* dst = &selectedIndexes[insertIndex])
                                    UnsafeUtility.MemMove(dst + 1, dst, (maxOutputBoxesPerClass - (insertIndex + 1)) * sizeof(int));
                                fixed (float* dst = &selectedScores[insertIndex])
                                    UnsafeUtility.MemMove(dst + 1, dst, (maxOutputBoxesPerClass - (insertIndex + 1)) * sizeof(int));
                            }
                        }

                        // record the score and index values of the selected box
                        // [x x x insertIndex y y y]
                        // insert box
                        // [x x x i y y y]
                        // [x x x score y y y]
                        selectedIndexes[insertIndex] = i;
                        selectedScores[insertIndex] = scores[batch, classID, i];
                        selectedBoxes = Mathf.Min(maxOutputBoxesPerClass, selectedBoxes + 1);
                    }

                    // gather outputs
                    for (int i = 0; i < selectedBoxes; i++)
                    {
                        // box is identified by its batch, class and index
                        outputData[numberOfBoxes * 3 + 0] = batch;
                        outputData[numberOfBoxes * 3 + 1] = classID;
                        outputData[numberOfBoxes * 3 + 2] = selectedIndexes[i];
                        numberOfBoxes++;
                    }
                }
            }

            // create output tensor of correct length by trimming outputData
            var O = ctx.backend.NewOutputTensorInt(new TensorShape(numberOfBoxes, 3));
            NativeTensorArray.Copy(outputData, ArrayTensorData.Pin(O, clearOnInit: false).array, numberOfBoxes * 3);
            return O;
        }

        bool NotIntersectOverUnion(float x1, float y1, float w1, float h1, float x2, float y2, float w2, float h2, Layers.CenterPointBox centerPointBox, float iouThreshold)
        {
            //inputs:
            //center_point_box:
            // 0 - diagonal y,x pairs (coords or normalized values)
            // 1 - center coords + width and height
            // Can be optimised by calculating each box area and PyTorch corner data outside IOU

            float b1x1;
            float b1x2;
            float b1y1;
            float b1y2;
            float b2x1;
            float b2x2;
            float b2y1;
            float b2y2;

            //convert inputs to: top left and bottom right corners of two rectangles
            //PyTorch
            if (centerPointBox == Layers.CenterPointBox.Center)
            {
                b1x1 = x1 - 0.5f * w1;
                b1x2 = x1 + 0.5f * w1;
                b1y1 = y1 - 0.5f * h1;
                b1y2 = y1 + 0.5f * h1;
                b2x1 = x2 - 0.5f * w2;
                b2x2 = x2 + 0.5f * w2;
                b2y1 = y2 - 0.5f * h2;
                b2y2 = y2 + 0.5f * h2;
            }
            //TensorFlow
            else //CenterPointBox.Corners
            {
                if (y1 < h1)
                {
                    b1x1 = y1;
                    b1x2 = h1;
                }
                else
                {
                    b1x1 = h1;
                    b1x2 = y1;
                }

                if (x1 < w1)
                {
                    b1y1 = x1;
                    b1y2 = w1;
                }
                else
                {
                    b1y1 = w1;
                    b1y2 = x1;
                }

                if (y2 < h2)
                {
                    b2x1 = y2;
                    b2x2 = h2;
                }
                else
                {
                    b2x1 = h2;
                    b2x2 = y2;
                }

                if (x2 < w2)
                {
                    b2y1 = x2;
                    b2y2 = w2;
                }
                else
                {
                    b2y1 = w2;
                    b2y2 = x2;
                }
            }

            //intersection rectangle
            float xMax = Math.Max(b1x1, b2x1);
            float yMax = Math.Max(b1y1, b2y1);
            float xMin = Math.Min(b1x2, b2x2);
            float yMin = Math.Min(b1y2, b2y2);

            //check if intersection rectangle exist
            if (xMin <= xMax || yMin <= yMax)
            {
                return true;
            }

            float intersectionArea = (xMin - xMax) * (yMin - yMax);
            float b1area = Math.Abs((b1x2 - b1x1) * (b1y2 - b1y1));
            float b2area = Math.Abs((b2x2 - b2x1) * (b2y2 - b2y1));
            return intersectionArea / (b1area + b2area - intersectionArea) <= iouThreshold;
        }

        internal override string profilerTag => "NonMaxSuppression";
    }

    /// <summary>
    /// Options for the pooling mode for `RoiAlign`.
    /// </summary>
    public enum RoiPoolingMode
    {
        /// <summary>
        /// Use average pooling.
        /// </summary>
        Avg = 0,
        /// <summary>
        /// Use maximum pooling.
        /// </summary>
        Max = 1
    }

    /// <summary>
    /// Represents an `RoiAlign` region of interest alignment layer. This calculates an output tensor by pooling the input tensor across each region of interest given by the `rois` tensor.
    /// </summary>
    [Serializable]
    public class RoiAlign : Layer
    {
        /// <summary>
        /// The pooling mode of the operation as an `RoiPoolingMode`.
        /// </summary>
        public RoiPoolingMode mode;
        /// <summary>
        /// The height of the output tensor.
        /// </summary>
        public int outputHeight;
        /// <summary>
        /// The width of the output tensor.
        /// </summary>
        public int outputWidth;
        /// <summary>
        /// The number of sampling points in the interpolation grid used to compute the output value of each pooled output bin.
        /// </summary>
        public int samplingRatio;
        /// <summary>
        /// The multiplicative spatial scale factor used to translate coordinates from their input spatial scale to the scale used when pooling.
        /// </summary>
        public float spatialScale;

        /// <summary>
        /// Initializes and returns an instance of `RoiAlign` region of interest alignment layer.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="input">The name to use for the input tensor of the layer.</param>
        /// <param name="rois">The name to use for the region of interest tensor of the layer.</param>
        /// <param name="batchIndices">The name to use for the 1D input tensor where each element denotes the index of the image in the batch for a given region of interest.</param>
        /// <param name="mode">The pooling mode of the operation as an `RoiPoolingMode`.</param>
        /// <param name="outputHeight">The height of the output tensor.</param>
        /// <param name="outputWidth">The width of the output tensor.</param>
        /// <param name="samplingRatio">The number of sampling points in the interpolation grid used to compute the output value of each pooled output bin.</param>
        /// <param name="spatialScale">The multiplicative spatial scale factor used to translate coordinates from their input spatial scale to the scale used when pooling.</param>
        public RoiAlign(string name, string input, string rois, string batchIndices, RoiPoolingMode mode, int outputHeight, int outputWidth, int samplingRatio, float spatialScale)
        {
            this.name = name;
            this.inputs = new[] { input, rois, batchIndices };
            this.mode = mode;
            this.outputHeight = outputHeight;
            this.outputWidth = outputWidth;
            this.samplingRatio = samplingRatio;
            this.spatialScale = spatialScale;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var shapeX = inputTensors[0].shape;
            var shapeRois = inputTensors[1].shape;
            var shapeIndices = inputTensors[2].shape;
            var shapeOut = SymbolicTensorShape.UnknownOfRank(4);

            shapeRois.DeclareRank(2);
            Logger.AssertIsFalse(shapeRois[1] != 4, "RoiAlign.ValueError: incorrect number of num_rois, expecting 4");
            shapeOut[0] = shapeRois[0];

            shapeX.DeclareRank(4);
            shapeOut[1] = shapeX[1];

            shapeIndices.DeclareRank(1);
            shapeOut[0] = SymbolicTensorDim.MaxDefinedDim(shapeOut[0], shapeIndices[0]);

            shapeOut[2] = new SymbolicTensorDim(outputHeight);
            shapeOut[3] = new SymbolicTensorDim(outputWidth);

            return new PartialTensor(DataType.Float, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var O = ctx.backend.NewOutputTensorFloat(ShapeInference.RoiAlign(inputTensors[0].shape, inputTensors[1].shape, inputTensors[2].shape, outputHeight, outputWidth));
            if (O.shape.HasZeroDims())
                return O;
            ctx.backend.RoiAlign(inputTensors[0] as TensorFloat, inputTensors[1] as TensorFloat, inputTensors[2] as TensorInt, O, mode, outputHeight, outputWidth, samplingRatio, spatialScale);
            return O;
        }

        /// <inheritdoc/>
        public override string ToString()
        {
            return $"{base.ToString()}, mode: {mode}, outputHeight: {outputHeight}, outputWidth: {outputWidth}, samplingRatio: {samplingRatio}, spatialScale: {spatialScale}";
        }

        internal override string profilerTag => "RoiAlign";
    }
}
