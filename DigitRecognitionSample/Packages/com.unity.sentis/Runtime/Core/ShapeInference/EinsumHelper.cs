using System;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

namespace Unity.Sentis
{
    static class EinsumHelper
    {
        /// calculates the strides for the output and sum to be used
        /// in the einsum backends
        public static unsafe void PinOperandStrides(TensorShape operandTensorShape, TensorIndex operandIndices, TensorIndex outputIndices, TensorIndex sumIndices, int* operandOutputStrides, int* operandSumStrides)
        {
            var operandShape = stackalloc int[8];
            var operandStrides = stackalloc int[8];
            OpsUtils.PinTensorShapeStrides(operandTensorShape, operandShape, operandStrides);

            UnsafeUtility.MemClear(operandOutputStrides, 8 * sizeof(int));
            UnsafeUtility.MemClear(operandSumStrides, 8 * sizeof(int));

            for (var k = 0; k < operandIndices.rank; k++)
            {
                for (var j = 0; j < sumIndices.rank; j++)
                {
                    if (sumIndices[j] == operandIndices[k])
                    {
                        operandSumStrides[8 - sumIndices.rank + j] += operandStrides[8 - operandIndices.rank + k];
                    }
                }

                for (var j = 0; j < outputIndices.rank; j++)
                {
                    if (outputIndices[j] == operandIndices[k])
                    {
                        operandOutputStrides[8 - outputIndices.rank + j] += operandStrides[8 - operandIndices.rank + k];
                    }
                }
            }
        }

        // given an einsum equation string and the operand shapes
        // calculates the output and sum indices and shapes
        public static void ParseEquationString(string equation, TensorShape[] operandShapes, ref TensorIndex[] operandIndices, out TensorIndex outputIndices, out TensorShape outputShape, out TensorIndex sumIndices, out TensorShape sumShape, out int numIndices)
        {
            unsafe
            {
                numIndices = 0;
                var isLabelMapped = stackalloc bool[128];
                var labelMapping = stackalloc int[128];

                var numOperands = 0;
                var numOperandLabels = stackalloc int[8];
                var operandStartIndices = stackalloc int[8];
                var operandEndIndices = stackalloc int[8];

                var outputStartIndex = -1;

                // first pass through equation, count and map labels
                // mark beginnings and ends of operands and output
                for (var i = 0; i < equation.Length; i++)
                {
                    var label = equation[i];

                    if (label > 0 && char.IsLetter(label))
                    {
                        if (!isLabelMapped[label])
                        {
                            isLabelMapped[label] = true;
                            labelMapping[label] = numIndices++;
                        }
                        if (outputStartIndex < 0)
                        {
                            numOperandLabels[numOperands]++;
                        }
                    }
                    else if (label == ',')
                    {
                        Logger.AssertIsTrue(numOperands < 8, "more than 8 operands are not supported in equation");
                        operandEndIndices[numOperands] = i;
                        numOperands++;
                        operandStartIndices[numOperands] = i+1;
                    }
                    else if (label == '-')
                    {
                        if (i + 1 < equation.Length && equation[i + 1] == '>')
                        {
                            Logger.AssertIsTrue(outputStartIndex < 0, "invalid einsum string");
                            operandEndIndices[numOperands] = i;
                            numOperands++;
                            outputStartIndex = i+2;
                        }
                    }
                }

                // if we didn't find the -> separator
                // mark the end of the operands
                if (outputStartIndex < 0)
                {
                    operandEndIndices[numOperands] = equation.Length;
                    numOperands++;
                }

                // calculate the number of broadcast dims
                var numBroadcastDims = 0;
                var startBroadcastDims = numIndices;
                for (var j = 0; j < numOperands; j++)
                {
                    numBroadcastDims = Mathf.Max(numBroadcastDims, operandShapes[j].rank - numOperandLabels[j]);
                }

                numIndices += numBroadcastDims;

                // counts for number of times each label is found
                // in operands and output
                var operandIndexCounts = stackalloc int[numIndices];
                var outputIndexCounts = stackalloc int[numIndices];
                var indexSize = stackalloc int[numIndices];

                // loop through sections of equation for each
                // operand, appending mapped index to list
                // and finding broadcasting ellipses
                for (var j = 0; j < numOperands; j++)
                {
                    var isFoundEllipsis = false;
                    operandIndices[j] = TensorIndex.Zeros(operandShapes[j].rank);
                    var rank = 0;

                    for (var i = operandStartIndices[j]; i < operandEndIndices[j]; i++)
                    {
                        var label = equation[i];

                        // A proper label for an axis
                        if (label > 0 && char.IsLetter(label))
                        {
                            var index = labelMapping[label];
                            operandIndexCounts[index]++;
                            operandIndices[j][rank] = index;

                            Logger.AssertIsTrue(indexSize[index] < 2 || operandShapes[j][rank] == 1 || operandShapes[j][rank] == indexSize[index], "axes with same label must have broadcastable sizes");
                            indexSize[index] = Mathf.Max(indexSize[index], operandShapes[j][rank]);

                            rank++;
                        } // The beginning of the ellipsis.
                        else if (label == '.')
                        {
                            // Check not already found ellipsis
                            Logger.AssertIsTrue(!isFoundEllipsis, "einstein sum subscripts string contains multiple ellipses");

                            // Check it's a proper ellipsis.
                            Logger.AssertIsTrue(i + 2 < operandEndIndices[j] && equation[++i] == '.' && equation[++i] == '.',"einstein sum subscripts string contains a '.' that is not part of an ellipsis ('...')");

                            for (var k = 0; k < numBroadcastDims; k++)
                            {
                                var index = startBroadcastDims + k;

                                operandIndexCounts[index]++;
                                operandIndices[j][rank] = index;

                                Logger.AssertIsTrue(indexSize[index] < 2 || operandShapes[j][rank] == 1 || operandShapes[j][rank] == indexSize[index], "axes with same label must have broadcastable sizes");
                                indexSize[index] = Mathf.Max(indexSize[index], operandShapes[j][rank]);

                                rank++;
                            }

                            isFoundEllipsis = true;
                        }
                        else
                        {
                            Logger.AssertIsTrue(label == ' ',$"invalid subscript {(char)label} in einstein sum subscripts string, subscripts must be letters");
                        }
                    }
                }

                var outputIndicesArray = stackalloc int[8];
                var outputRank = 0;

                if (outputStartIndex >= 0)
                {
                    // explicit mode: loop through section of equation for
                    // output, appending mapped index to list
                    // and finding broadcasting ellipses
                    var isFoundEllipsis = false;
                    for (var i = outputStartIndex; i < equation.Length; i++)
                    {
                        var label = equation[i];

                        // a proper label for an axis
                        if (label > 0 && char.IsLetter(label))
                        {
                            var index = labelMapping[label];
                            Logger.AssertIsTrue(outputIndexCounts[index] < 1,"output label may not appear more than once in equation string");
                            Logger.AssertIsTrue(operandIndexCounts[index] > 0, "label in output must appear in operands");
                            outputIndicesArray[outputRank++] = index;
                            outputIndexCounts[index]++;
                        } // the beginning of the ellipsis.
                        else if (label == '.')
                        {
                            // check not already found ellipsis
                            Logger.AssertIsTrue(!isFoundEllipsis, "einstein sum subscripts string contains multiple ellipses");

                            // check it's a proper ellipsis.
                            Logger.AssertIsTrue(i + 2 < equation.Length && equation[++i] == '.' && equation[++i] == '.',"einstein sum subscripts string contains a '.' that is not part of an ellipsis ('...')");

                            for (var k = 0; k < numBroadcastDims; k++)
                            {
                                var index = startBroadcastDims + k;
                                outputIndicesArray[outputRank++] = index;
                                outputIndexCounts[index]++;
                            }
                            isFoundEllipsis = true;
                        }
                        else
                        {
                            Logger.AssertIsTrue(label == ' ',$"invalid subscript {(char)label} in einstein sum subscripts string, subscripts must be letters");
                        }
                    }
                }
                else
                {
                    // implicit mode, add labels in alphabetical order
                    // with broadcast first (if somewhere in operands)

                    for (var k = 0; k < numBroadcastDims; k++)
                    {
                        var index = startBroadcastDims + k;
                        outputIndicesArray[outputRank++] = index;
                        outputIndexCounts[index]++;
                    }

                    for (var i = 0; i < numIndices; i++)
                    {
                        if (operandIndexCounts[i] == 1)
                        {
                            outputIndicesArray[outputRank++] = i;
                            outputIndexCounts[i]++;
                        }
                    }
                }

                // calculate the output indices and shape
                outputIndices = TensorIndex.Zeros(outputRank);
                outputShape = TensorShape.Ones(outputRank);

                for (var i = 0; i < outputRank; i++)
                {
                    outputIndices[i] = outputIndicesArray[i];
                    outputShape[i] = indexSize[outputIndices[i]];
                }

                // calculate the sum indices and shape
                // i.e. those not in the output
                sumIndices = TensorIndex.Zeros(numIndices - outputRank);
                sumShape = TensorShape.Ones(numIndices - outputRank);

                var sumRank = 0;
                for (var i = 0; i < numIndices; i++)
                {
                    if (outputIndexCounts[i] == 0)
                    {
                        sumIndices[sumRank] = i;
                        sumShape[sumRank] = indexSize[i];
                        sumRank++;
                    }
                }
            }
        }

        // given an einsum equation string and the operand shapes
        // calculates the static output tensor shape
        public static SymbolicTensorShape ParseEquationStringShape(string equation, SymbolicTensorShape[] operandShapes, ref TensorIndex[] operandIndices, out TensorIndex outputIndices, out int numIndices)
        {
            unsafe
            {
                numIndices = 0;
                var isLabelMapped = stackalloc bool[128];
                var labelMapping = stackalloc int[128];

                var numOperands = 0;
                var numOperandLabels = stackalloc int[8];
                var isOperandEllipses = stackalloc bool[8];
                var operandStartIndices = stackalloc int[8];
                var operandEndIndices = stackalloc int[8];

                var outputStartIndex = -1;

                // first pass through equation, count and map labels
                // mark beginnings and ends of operands and output
                for (var i = 0; i < equation.Length; i++)
                {
                    var label = equation[i];

                    if (label > 0 && char.IsLetter(label))
                    {
                        if (!isLabelMapped[label])
                        {
                            isLabelMapped[label] = true;
                            labelMapping[label] = numIndices++;
                        }
                        if (outputStartIndex < 0)
                        {
                            numOperandLabels[numOperands]++;
                        }
                    }
                    else if (label == ',')
                    {
                        Logger.AssertIsTrue(numOperands < 8, "more than 8 operands are not supported in equation");
                        operandEndIndices[numOperands] = i;
                        numOperands++;
                        operandStartIndices[numOperands] = i+1;
                    }
                    else if (label == '-')
                    {
                        if (i + 1 < equation.Length && equation[i + 1] == '>')
                        {
                            Logger.AssertIsTrue(outputStartIndex < 0, "invalid einsum string");
                            operandEndIndices[numOperands] = i;
                            numOperands++;
                            outputStartIndex = i+2;
                        }
                    }
                    else if (label == '.')
                    {
                        isOperandEllipses[numOperands] = true;
                    }
                }

                // if we didn't find the -> separator
                // mark the end of the operands
                if (outputStartIndex < 0)
                {
                    operandEndIndices[numOperands] = equation.Length;
                    numOperands++;
                }

                // calculate the number of broadcast dims
                var isBroadcastOperands = false;
                var isBroadcastOperandsUnknown = true;
                var numBroadcastDims = 0;
                var startBroadcastDims = numIndices;
                for (var j = 0; j < numOperands; j++)
                {
                    if (isOperandEllipses[j])
                    {
                        isBroadcastOperands = true;
                        if (operandShapes[j].hasRank)
                        {
                            isBroadcastOperandsUnknown = false;
                            numBroadcastDims = Mathf.Max(numBroadcastDims, operandShapes[j].rank - numOperandLabels[j]);
                        }
                    }
                }

                if (isBroadcastOperands && isBroadcastOperandsUnknown)
                {
                    outputIndices = default;
                    return SymbolicTensorShape.UnknownShape;
                }

                Logger.AssertIsTrue(operandShapes.Length == numOperands, "number of operand shapes must equal the number of operands in equation");

                for (var j = 0; j < numOperands; j++)
                {
                    var operandRank = isOperandEllipses[j] ? numOperandLabels[j] + numBroadcastDims : numOperandLabels[j];
                    if (!operandShapes[j].hasRank)
                    {
                        operandShapes[j] = SymbolicTensorShape.UnknownOfRank(operandRank);
                    }

                    Logger.AssertIsTrue(operandShapes[j].rank == operandRank, "operand rank must match that from equation");
                }

                numIndices += numBroadcastDims;

                // counts for number of times each label is found
                // in operands and output
                operandIndices = new TensorIndex[numOperands];
                var operandIndexCounts = stackalloc int[numIndices];
                var outputIndexCounts = stackalloc int[numIndices];
                var indexSize = new SymbolicTensorDim[numIndices];

                for (var i = 0; i < numIndices; i++)
                {
                    indexSize[i] = SymbolicTensorDim.One;
                }

                // loop through sections of equation for each
                // operand, appending mapped index to list
                // and finding broadcasting ellipses
                for (var j = 0; j < numOperands; j++)
                {
                    var isFoundEllipsis = false;
                    operandIndices[j] = TensorIndex.Zeros(operandShapes[j].rank);
                    var rank = 0;

                    for (var i = operandStartIndices[j]; i < operandEndIndices[j]; i++)
                    {
                        var label = equation[i];

                        // A proper label for an axis
                        if (label > 0 && char.IsLetter(label))
                        {
                            var index = labelMapping[label];
                            operandIndexCounts[index]++;
                            operandIndices[j][rank] = index;

                            indexSize[index] = indexSize[index].Broadcast(operandShapes[j][rank]);

                            rank++;
                        } // The beginning of the ellipsis.
                        else if (label == '.')
                        {
                            // Check not already found ellipsis
                            Logger.AssertIsTrue(!isFoundEllipsis, "einstein sum subscripts string contains multiple ellipses");

                            // Check it's a proper ellipsis.
                            Logger.AssertIsTrue(i + 2 < operandEndIndices[j] && equation[++i] == '.' && equation[++i] == '.',"einstein sum subscripts string contains a '.' that is not part of an ellipsis ('...')");

                            for (var k = 0; k < numBroadcastDims; k++)
                            {
                                var index = startBroadcastDims + k;

                                operandIndexCounts[index]++;
                                operandIndices[j][rank] = index;

                                indexSize[index] = indexSize[index].Broadcast(operandShapes[j][rank]);

                                rank++;
                            }

                            isFoundEllipsis = true;
                        }
                        else
                        {
                            Logger.AssertIsTrue(label == ' ',$"invalid subscript {(char)label} in einstein sum subscripts string, subscripts must be letters");
                        }
                    }
                }

                var outputIndicesArray = stackalloc int[8];
                var outputRank = 0;

                if (outputStartIndex >= 0)
                {
                    // explicit mode: loop through section of equation for
                    // output, appending mapped index to list
                    // and finding broadcasting ellipses
                    var isFoundEllipsis = false;
                    for (var i = outputStartIndex; i < equation.Length; i++)
                    {
                        var label = equation[i];

                        // a proper label for an axis
                        if (label > 0 && char.IsLetter(label))
                        {
                            var index = labelMapping[label];
                            Logger.AssertIsTrue(outputIndexCounts[index] < 1,"output label may not appear more than once in equation string");
                            Logger.AssertIsTrue(operandIndexCounts[index] > 0, "label in output must appear in operands");
                            outputIndicesArray[outputRank++] = index;
                            outputIndexCounts[index]++;
                        } // the beginning of the ellipsis.
                        else if (label == '.')
                        {
                            // check not already found ellipsis
                            Logger.AssertIsTrue(!isFoundEllipsis, "einstein sum subscripts string contains multiple ellipses");

                            // check it's a proper ellipsis.
                            Logger.AssertIsTrue(i + 2 < equation.Length && equation[++i] == '.' && equation[++i] == '.',"einstein sum subscripts string contains a '.' that is not part of an ellipsis ('...')");

                            for (var k = 0; k < numBroadcastDims; k++)
                            {
                                var index = startBroadcastDims + k;
                                outputIndicesArray[outputRank++] = index;
                                outputIndexCounts[index]++;
                            }
                            isFoundEllipsis = true;
                        }
                        else
                        {
                            Logger.AssertIsTrue(label == ' ',$"invalid subscript {(char)label} in einstein sum subscripts string, subscripts must be letters");
                        }
                    }
                }
                else
                {
                    // implicit mode, add labels in alphabetical order
                    // with broadcast first (if somewhere in operands)

                    for (var k = 0; k < numBroadcastDims; k++)
                    {
                        var index = startBroadcastDims + k;
                        outputIndicesArray[outputRank++] = index;
                        outputIndexCounts[index]++;
                    }

                    for (var i = 0; i < numIndices; i++)
                    {
                        if (operandIndexCounts[i] == 1)
                        {
                            outputIndicesArray[outputRank++] = i;
                            outputIndexCounts[i]++;
                        }
                    }
                }

                // calculate the output indices and shape
                var outputDims = SymbolicTensorShape.UnknownOfRank(outputRank);
                outputIndices = TensorIndex.Zeros(outputRank);

                for (var i = 0; i < outputRank; i++)
                {
                    outputIndices[i] = outputIndicesArray[i];
                    outputDims[i] = indexSize[outputIndicesArray[i]];
                }

                return outputDims;
            }
        }
    }
}
