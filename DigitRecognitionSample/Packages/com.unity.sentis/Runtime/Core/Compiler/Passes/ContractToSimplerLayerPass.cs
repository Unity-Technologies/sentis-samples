using System.Linq;
using Unity.Sentis.Compiler.Analyser;
using System.Collections.Generic;
using System;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class ContractToSimplerLayerPass : IModelPass
    {
        public void Run(ref Model model)
        {
            using var op = new CPUOps();
            var ctx = PartialInferenceAnalysis.InferModelPartialTensors(model, true);

            var modelConstants = new Dictionary<string, Layers.Constant>();
            foreach (var c in model.constants)
                modelConstants.Add(c.name, c);

            for (int l = 0; l < model.layers.Count; ++l)
            {
                var layer = model.layers[l];

                if (layer is Layers.Concat concatLayer)
                {
                    // replace concat with one input with identity
                    if (layer.inputs.Length == 1)
                    {
                        model.layers[l] = new Layers.Identity(concatLayer.name, concatLayer.inputs[0]);
                        continue;
                    }

                    // replace Concat layer which concatenates the only same tensor multiple times with Tile layer
                    var shape = ctx.GetPartialTensor(concatLayer.name).shape;
                    if (!shape.hasRank || concatLayer.inputs.Any(o => o != layer.inputs[0]))
                        continue;

                    var tileShape = TensorShape.Ones(shape.rank);
                    tileShape[concatLayer.axis] = concatLayer.inputs.Length;

                    var repeatsConstant = new Layers.Constant(model.GetUniqueName(concatLayer.name + "_Repeats"), new TensorInt(new TensorShape(tileShape.rank), tileShape.ToArray()));
                    model.AddConstant(repeatsConstant);
                    model.layers[l] = new Layers.Tile(concatLayer.name, concatLayer.inputs[0], repeatsConstant.name);
                    continue;
                }
                if (layer is Layers.Transpose transposeLayer)
                {
                    // replace Transpose layer which does not actually transpose the dimensions with the identity
                    // TODO: can add cases such as tensor.shape = (1, 1, 4) permutation = [1, 0, 2]
                    if (transposeLayer.permutations == null)
                        continue;

                    bool nopTranspose = true;
                    for (int i = 0; i < transposeLayer.permutations.Length; ++i)
                    {
                        if (transposeLayer.permutations[i] == i)
                            continue;
                        nopTranspose = false;
                        break;
                    }

                    if (nopTranspose)
                        model.layers[l] = new Layers.Identity(transposeLayer.name, transposeLayer.inputs[0]);
                    continue;
                }
                if (layer is Layers.Reshape reshapeLayer)
                {
                    // replace Reshape layer which does not reshape with identity layer
                    var outputShape = ctx.GetPartialTensor(reshapeLayer.name).shape;
                    var inputShape = ctx.GetPartialTensor(reshapeLayer.inputs[0]).shape;
                    if (!outputShape.hasRank || !inputShape.hasRank || outputShape.rank != inputShape.rank)
                        continue;

                    var shapeTensor = ctx.GetPartialTensor(reshapeLayer.inputs[1]);

                    var isIdentity = true;
                    for (var i = 0; i < outputShape.rank && isIdentity; i++)
                    {
                        if (!(outputShape[i] == inputShape[i]) && !(reshapeLayer.allowZero && shapeTensor[i] == 0))
                            isIdentity = false;
                    }

                    if (isIdentity)
                        model.layers[l] = new Layers.Identity(layer.name, layer.inputs[0]);
                    continue;
                }
                if (layer is Layers.Expand expandLayer)
                {
                    // replace Expand layer which does not change the number of elements with a reshape or identity layer
                    // TODO: add support for cases such as tensor.shape = (A), expandShape = [1, A] which is a Reshape
                    var outputShape = ctx.GetPartialTensor(expandLayer.name).shape;
                    var inputShape = ctx.GetPartialTensor(expandLayer.inputs[0]).shape;
                    if (!outputShape.hasRank || !inputShape.hasRank)
                        continue;

                    var shapeTensor = ctx.GetPartialTensor(expandLayer.inputs[1]);

                    if (outputShape.rank == inputShape.rank)
                    {
                        var isIdentity = true;
                        for (var i = 0; i < outputShape.rank && isIdentity; i++)
                        {
                            if (i < shapeTensor.length && shapeTensor[i] == 1)
                                continue;
                            if (!(outputShape[i] == inputShape[i]))
                                isIdentity = false;
                        }

                        if (isIdentity)
                        {
                            model.layers[l] = new Layers.Identity(layer.name, layer.inputs[0]);
                            continue;
                        }
                    }

                    if (outputShape.IsFullyKnown() && inputShape.IsFullyKnown() && outputShape.Length() == inputShape.Length())
                    {
                        var shapeName = model.GetUniqueName(layer.name + "_Shape");
                        var shapeConstant = new Layers.Constant(model.GetUniqueName(shapeName), new TensorInt(new TensorShape(outputShape.rank), outputShape.ToTensorShape().ToArray()));
                        model.AddConstant(shapeConstant);
                        model.layers[l] = new Layers.Reshape(layer.name, layer.inputs[0], shapeConstant.name);
                    }

                    continue;
                }
                if (layer is Layers.Tile tileLayer)
                {
                    // replace Tile layer where the tile values are all 1 with Identity
                    var repeats = ctx.GetPartialTensor(tileLayer.inputs[1]);

                    if (!repeats.IsFullyKnown())
                        continue;
                    var allOnes = true;
                    for (var i = 0; i < repeats.length; i++)
                    {
                        allOnes &= repeats[i] == 1;
                    }
                    if (!allOnes)
                        continue;

                    model.layers[l] = new Layers.Identity(tileLayer.name, tileLayer.inputs[0]);
                    continue;
                }
                if (layer is Layers.Reduce reduceLayer)
                {
                    // replace Reduce layer which does not perform any reduction with Identity
                    var axes = reduceLayer.inputs.Length > 1 ? ctx.GetPartialTensor(reduceLayer.inputs[1]) : null;
                    var isEmptyAxes = (axes == null || axes.shape.Length() == 0);
                    if (reduceLayer.noopWithEmptyAxes && isEmptyAxes)
                    {
                        model.layers[l] = new Layers.Identity(reduceLayer.name, reduceLayer.inputs[0]);
                        continue;
                    }

                    // these reductions don't change the values if we reduce along a 0 or 1 length axis
                    if (isEmptyAxes || !axes.IsFullyKnown() || !(reduceLayer is Layers.ReduceMin or Layers.ReduceMax || reduceLayer is Layers.ReduceSum || reduceLayer is Layers.ReduceLogSumExp || reduceLayer is Layers.ReduceProd))
                        continue;

                    var input = ctx.GetPartialTensor(reduceLayer.inputs[0]);
                    if (!input.shape.hasRank)
                        continue;

                    var isTrivialReduction = true;
                    for (var i = 0; i < axes.length && isTrivialReduction; i++)
                    {
                        if (input.shape[axes[i].intValue] <= 1)
                            continue;
                        isTrivialReduction = false;
                    }

                    if (!isTrivialReduction)
                        continue;

                    if (reduceLayer.keepdims)
                        model.layers[l] = new Layers.Identity(reduceLayer.name, reduceLayer.inputs[0]);
                    else
                        model.layers[l] = new Layers.Squeeze(reduceLayer.name, reduceLayer.inputs[0], reduceLayer.inputs[1]);
                    continue;
                }
                if (layer is Layers.Cast castLayer)
                {
                    // replace Cast which casts to same type as input with Identity
                    var input = ctx.GetPartialTensor(castLayer.inputs[0]);
                    if (input.dataType != castLayer.toType)
                        continue;

                    model.layers[l] = new Layers.Identity(castLayer.name, castLayer.inputs[0]);
                    continue;
                }
                if (layer is Layers.CastLike castLikeLayer)
                {
                    // replace CastLike with Cast or Identity
                    var input = ctx.GetPartialTensor(castLikeLayer.inputs[0]);
                    var targetType = ctx.GetPartialTensor(castLikeLayer.inputs[1]);
                    if (input.dataType == targetType.dataType)
                    {
                        model.layers[l] = new Layers.Identity(castLikeLayer.name, castLikeLayer.inputs[0]);
                        continue;
                    }

                    model.layers[l] = new Layers.Cast(castLikeLayer.name, castLikeLayer.inputs[0], targetType.dataType);
                    continue;
                }
                if (layer is Layers.BatchNormalization bnLayer)
                {
                    bool allConstInputs = true;
                    for (int i = 1; i < bnLayer.inputs.Length; i++)
                    {
                        allConstInputs &= modelConstants.ContainsKey(bnLayer.inputs[i]);
                    }
                    if (!allConstInputs)
                        continue;

                    var c1 = modelConstants[bnLayer.inputs[1]];
                    var c2 = modelConstants[bnLayer.inputs[2]];
                    var c3 = modelConstants[bnLayer.inputs[3]];
                    var c4 = modelConstants[bnLayer.inputs[4]];

                    using var gamma = c1.DataSetToTensorView() as TensorFloat;
                    using var beta = c2.DataSetToTensorView() as TensorFloat;
                    using var mean = c3.DataSetToTensorView() as TensorFloat;
                    using var variance = c4.DataSetToTensorView() as TensorFloat;

                    var epsilonTensor = op.ConstantOfShape(new TensorShape(1), bnLayer.epsilon);
                    var sqrtVar = op.Sqrt(op.Add(variance, epsilonTensor));
                    var scale = op.Div(gamma, sqrtVar);
                    var bias = op.Sub(beta, op.Mul(scale, mean));

                    var scaleConstantName = model.GetUniqueName($"{bnLayer.name}_Scale");
                    var biasConstantName = model.GetUniqueName($"{bnLayer.name}_Bias");

                    model.AddConstant(new Layers.Constant(scaleConstantName, scale));
                    model.AddConstant(new Layers.Constant(biasConstantName, bias));

                    model.layers[l] = new Layers.ScaleBias(bnLayer.name, bnLayer.inputs[0], scaleConstantName, biasConstantName);
                    continue;
                }
            }
        }
    }
}
