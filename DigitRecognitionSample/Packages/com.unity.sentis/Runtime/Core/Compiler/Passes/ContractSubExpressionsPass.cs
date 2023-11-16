using System.Linq;
using Unity.Sentis.Compiler.Analyser;
using Unity.Sentis.Layers;
using Unity.Sentis;
using System.Collections.Generic;
using System;
using UnityEngine;

namespace Unity.Sentis.Compiler.Passes.Optimization
{
    class ContractSubExpressionPass : IModelPass
    {
        Dictionary<string, Constant> modelConstants = new Dictionary<string, Constant>();
        Dictionary<string, Layer> nameToLayer = new Dictionary<string, Layer>();
        Dictionary<string, int> nameToIndex = new Dictionary<string, int>();
        Dictionary<string, List<Layer>> downstreamLayers = new Dictionary<string, List<Layer>>();
        List<Layer> layersInPattern = new List<Layer>();
        List<string> inputLayers = new List<string>();
        List<Constant> inputConstants = new List<Constant>();

        // Automatic Chain rule CAS
        // we construct a graph of successive operations by chaining operators and function calls
        //  (x + 2) / 3 + x
        // that creates a graph of INode storing each operations
        // we can call .Validate on that object and it will recursively walk the graph and check validity if layer inputs match the subgraph
        // leaf node:
        //  * constant, match if constant exists and has the expected value
        //  * input, always match
        // tree node:
        //  * check layer type matches INode type
        //  * returns true if all inputs are valid
        private abstract class INode
        {
            public static INode operator *(INode a, INode b)
            {
                return new LayerNode<Mul>(a, b);
            }
            public static INode operator *(INode a, float b)
            {
                return new LayerNode<Mul>(a, b);
            }
            public static INode operator +(INode a, INode b)
            {
                return new LayerNode<Add>(a, b);
            }
            public static INode operator +(INode a, float b)
            {
                return new LayerNode<Add>(a, b);
            }
            public static INode operator -(INode a, INode b)
            {
                return new LayerNode<Sub>(a, b);
            }
            public static INode operator /(INode a, INode b)
            {
                return new LayerNode<Div>(a, b);
            }
            public static INode operator /(INode a, float b)
            {
                return new LayerNode<Div>(a, b);
            }
            public static INode Erf(INode a)
            {
                return new LayerNode<Erf>(a);
            }
            public static INode Sigmoid(INode a)
            {
                return new LayerNode<Sigmoid>(a);
            }
            public static INode Softmax(INode a, int b)
            {
                return new LayerNode<Softmax>(a, b);
            }
            public static INode Transpose(INode a, int[] b)
            {
                return new LayerNode<Transpose>(a, b);
            }
            public static INode Reshape(INode a, INode b)
            {
                return new LayerNode<Reshape>(a, b);
            }
            public static INode MatMul(INode a, INode b)
            {
                return new LayerNode<MatMul>(a, b);
            }
            public static INode MatMul2D(INode a, INode b)
            {
                return new LayerNode<MatMul2D>(a, b);
            }
            public static INode Pow(INode a, float b)
            {
                return new LayerNode<Pow>(a, b);
            }
            public static INode Sqrt(INode a)
            {
                return new LayerNode<Sqrt>(a);
            }
            public static INode ReduceMean(INode a, int b)
            {
                return new LayerNode<ReduceMean>(a, b);
            }
        }

        private class InputNode : INode
        {
        }

        abstract class IConstantNode : INode
        {
            public abstract bool Validate(Constant constant);
        }

        private class ConstantFloatTensor : IConstantNode
        {
            public override bool Validate(Constant constant)
            {
                if (constant.dataType != DataType.Int)
                    return false;

                return true;
            }
        }

        class ScalarInt : IConstantNode
        {
            float m_Value;
            public ScalarInt(int v)
            {
                m_Value = v;
            }

            public override bool Validate(Constant constant)
            {
                if (constant.dataType != DataType.Int)
                    return false;

                return constant.length == 1 && constant.shape.rank <= 1 && constant.weights.Get<int>(0) == m_Value;
            }
        }

        class ScalarFloat : IConstantNode
        {
            float m_Value;
            public ScalarFloat(float v)
            {
                m_Value = v;
            }

            public override bool Validate(Constant constant)
            {
                if (constant.dataType != DataType.Float)
                    return false;

                return constant.length == 1 && constant.shape.rank <= 1 && constant.weights.Get<float>(0) == m_Value;
            }
        }

        class VariableScalarFloat : IConstantNode
        {
            public override bool Validate(Constant constant)
            {
                if (constant.dataType != DataType.Float)
                    return false;

                return constant.length == 1 && constant.shape.rank <= 1;
            }
        }

        class VectorInt : IConstantNode
        {
            int[] m_Value;
            public VectorInt(int[] v)
            {
                m_Value = v;
            }

            public override bool Validate(Constant constant)
            {
                if (constant.dataType != DataType.Int)
                    return false;

                if (constant.length != m_Value.Length)
                    return false;

                for (int i = 0; i < m_Value.Length; i++)
                {
                    if (constant.weights.Get<int>(i) != m_Value[i])
                        return false;
                }

                return true;
            }
        }

        abstract class ILayerNode : INode
        {
            public INode[] inputs;
            public abstract bool Validate(Layer layer);
        }

        class LayerNode<T> : ILayerNode where T : Layer
        {
            public LayerNode(INode i0)
            {
                inputs = new[] { i0 };
            }
            public LayerNode(INode i0, INode i1)
            {
                inputs = new[] { i0, i1 };
            }
            public LayerNode(INode i0, float i1)
            {
                inputs = new[] { i0, new ScalarFloat(i1) };
            }
            public LayerNode(INode i0, int i1)
            {
                inputs = new[] { i0, new ScalarInt(i1) };
            }
            public LayerNode(INode i0, int[] i1)
            {
                inputs = new[] { i0, new VectorInt(i1) };
            }

            public override bool Validate(Layer layer)
            {
                return layer is T;
            }
        }

        // remapping rules:
        // key: expression to test against
        // value: layer to spawn, layersInPattern is all the layers that match the expression
        Dictionary<Func<InputNode, INode>, Func<Layer, List<string>, List<Constant>, Layer>> remappingRules = new Dictionary<Func<InputNode, INode>, Func<Layer, List<string>, List<Constant>, Layer>>()
        {
            { x => INode.Pow(x, -1.0f),                                      (y, iLayers, iConstants) => new Reciprocal(y.name, iLayers[0]) },
            { x => INode.Pow(x, 0.5f),                                       (y, iLayers, iConstants) => new Sqrt(y.name, iLayers[0]) },
            { x => INode.Pow(x, 1.0f),                                       (y, iLayers, iConstants) => new Identity(y.name, iLayers[0]) },
            { x => INode.Pow(x, 2.0f),                                       (y, iLayers, iConstants) => new Square(y.name, iLayers[0]) },
            { x => (x * INode.Sigmoid(x)),                                   (y, iLayers, iConstants) => new Swish(y.name, iLayers[0]) },
            { x => (x * (INode.Erf((x / Mathf.Sqrt(2.0f))) + 1.0f)) * 0.5f,  (y, iLayers, iConstants) => new Gelu(y.name, iLayers[0]) },
            { x => {
                var mean = INode.ReduceMean(x, -1);
                var y = x - mean;
                var variance = INode.ReduceMean(INode.Pow(y, 2.0f), -1);
                var epsilon = new VariableScalarFloat();
                var v = y / INode.Sqrt(variance + epsilon);
                var scale = new InputNode();
                var bias = new InputNode();
                return v * scale + bias; },
                    (y, iLayers, iConstants) => {
                    float epsilon = iConstants[0].weights.Get<float>(0);
                    return new LayerNormalization(y.name, iLayers[iLayers.Count - 1], iLayers[1], iLayers[0], epsilon);
                }
            },
            { x => {
                var q_weights = new InputNode();
                var q_bias = new InputNode();

                var k_weights = new InputNode();
                var k_bias = new InputNode();

                var v_weights = new InputNode();
                var v_bias = new InputNode();

                var dim = new InputNode();
                var attn_output_dim = new InputNode();
            
                var scaling_factor = new VariableScalarFloat();

                var o_weights = new InputNode();
                var o_bias = new InputNode();

                var q = INode.MatMul(x, q_weights);
                var q_add =  q_bias + q;
                var q_reshape = INode.Reshape(q_add, dim);
                var q_transpose = INode.Transpose(q_reshape, new[] {0, 2, 1, 3});
           
                var k = INode.MatMul(x, k_weights);
                var k_add = k_bias + k;
                var k_reshape = INode.Reshape(k_add, dim);
                var k_transpose = INode.Transpose(k_reshape, new[] {0, 2, 3, 1});
           
                var v = INode.MatMul(x, v_weights);
                var v_add = v_bias + v;
                var v_reshape = INode.Reshape(v_add, dim);
                var v_transpose = INode.Transpose(v_reshape, new[] {0, 2, 3, 1});
           
                var qk = INode.MatMul(q_transpose, k_transpose);
                var qk_scale = qk * scaling_factor;
                var attn_output_weights = INode.Softmax(qk_scale, -1);
                var attn_output = INode.MatMul(attn_output_weights, v_transpose);
                var attn_output_transpose = INode.Transpose(attn_output, new[] {0, 2, 3, 1});
                var attn_output_reshape = INode.Reshape(attn_output_transpose, attn_output_dim);
                var atten_output_gemm = INode.MatMul(attn_output_reshape, o_weights);
                var atten_output_final = o_bias + atten_output_gemm;
                return atten_output_final;
            }, (y, iLayers, iConstants) => {
                float scaling_factor = iConstants[0].weights.Get<float>(0);
                return new SingleHeadAttention(y.name, iLayers[12], iLayers[11], iLayers[13], iLayers[7], iLayers[9], iLayers[3], iLayers[5], iLayers[0], iLayers[14], scaling_factor); }
            }, // "x", "q_weights", "y", "k_weights", "z", "v_weights", "out_proj_weight", "out_proj_bias", "num_heads"
            { x => x + new VariableScalarFloat(), (y, iLayers, iConstants) => new ScalarMad(y.name, iLayers[0], 1.0f, iConstants[0].weights.Get<float>(0)) },
            { x => new VariableScalarFloat() + x, (y, iLayers, iConstants) => new ScalarMad(y.name, iLayers[0], 1.0f, iConstants[0].weights.Get<float>(0)) },
            { x => x - new VariableScalarFloat(), (y, iLayers, iConstants) => new ScalarMad(y.name, iLayers[0], 1.0f, -iConstants[0].weights.Get<float>(0)) },
            { x => new VariableScalarFloat() - x, (y, iLayers, iConstants) => new ScalarMad(y.name, iLayers[0], -1.0f, iConstants[0].weights.Get<float>(0)) },
            { x => x * new VariableScalarFloat(), (y, iLayers, iConstants) => new ScalarMad(y.name, iLayers[0], iConstants[0].weights.Get<float>(0), 0) },
            { x => new VariableScalarFloat() * x, (y, iLayers, iConstants) => new ScalarMad(y.name, iLayers[0], iConstants[0].weights.Get<float>(0), 0) },
            { x => x / new VariableScalarFloat(), (y, iLayers, iConstants) => new ScalarMad(y.name, iLayers[0], 1.0f / iConstants[0].weights.Get<float>(0), 0) },
        };

        bool Validate(INode root, Layer input)
        {
            Stack<string> layerStack = new Stack<string>();
            Stack<INode> nodeStack = new Stack<INode>();

            nodeStack.Push(root);
            layerStack.Push(input.name);
            while (nodeStack.Count != 0)
            {
                INode node = nodeStack.Pop();
                string name = layerStack.Pop();

                if (node is IConstantNode cNode)
                {
                    if (!modelConstants.TryGetValue(name, out Constant constant))
                        return false;

                    if (!cNode.Validate(constant))
                        return false;

                    inputConstants.Add(constant);
                }
                else if (node is ILayerNode lNode)
                {
                    if (!nameToLayer.TryGetValue(name, out Layer layer))
                        return false;

                    if (!lNode.Validate(layer))
                        return false;

                    layersInPattern.Add(layer);
                    for (int i = 0; i < layer.inputs.Length; i++)
                    {
                        layerStack.Push(layer.inputs[i]);
                        nodeStack.Push(lNode.inputs[i]);
                    }
                }
                else if (node is InputNode)
                {
                    inputLayers.Add(name);
                }
            }

            return true;
        }

        // Pattern is said to be fully included if
        // foreach layer in subgraph
        // * all input are in the subGraph or inputs of root layer
        // * all downstream layers are in the subGraph or downstream of root layer
        bool CheckGraphFullInclusion(Layer root, List<Layer> subGraph, out string patternInput)
        {
            patternInput = null;
            HashSet<string> layersInSubGraph = new HashSet<string>();
            foreach (var layer in subGraph)
            {
                layersInSubGraph.Add(layer.name);
                foreach (var input in layer.inputs)
                {
                    layersInSubGraph.Add(input);
                }
                foreach (var downStream in downstreamLayers[layer.name])
                {
                    if (downStream != null)
                        layersInSubGraph.Add(downStream.name);
                }
            }

            foreach (var layer in layersInPattern)
            {
                layersInSubGraph.Remove(layer.name);
                foreach (var input in layer.inputs)
                {
                    if (modelConstants.ContainsKey(input))
                        layersInSubGraph.Remove(input);
                }
            }

            foreach (var downStream in downstreamLayers[root.name])
            {
                if (downStream == null)
                    continue;
                if (!layersInSubGraph.Contains(downStream.name))
                    return false;
                layersInSubGraph.Remove(downStream.name);
            }

            if (layersInSubGraph.Count != 1)
                return false;

            patternInput = layersInSubGraph.ElementAt(0);

            return true;
        }

        public void Run(ref Model model)
        {
            HashSet<string> layersToRemove = new HashSet<string>();

            // build static helpers:
            // - name -> constant
            // - name -> layer index
            modelConstants = new Dictionary<string, Constant>();
            foreach (var c in model.constants)
                modelConstants.Add(c.name, c);

            nameToLayer = new Dictionary<string, Layer>();
            nameToIndex = new Dictionary<string, int>();
            downstreamLayers = new Dictionary<string, List<Layer>>();
            var outputs = new HashSet<string>();
            foreach (var o in model.outputs)
                outputs.Add(o);
            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];
                nameToLayer.Add(layer.name, layer);
                nameToIndex.Add(layer.name, l);

                foreach (var input in layer.inputs)
                {
                    if (downstreamLayers.ContainsKey(input))
                        downstreamLayers[input].Add(layer);
                    else
                        downstreamLayers[input] = new List<Layer> { layer };
                }

                if (outputs.Contains(layer.name))
                {
                    downstreamLayers[layer.name] = new List<Layer> { null };
                }
            }

            var x = new InputNode();
            layersInPattern = new List<Layer>();
            inputLayers = new List<string>();
            inputConstants = new List<Constant>();

            // Algorithm:
            // foreach layers
            //  foreach pattern
            //      check if pattern is matched walking up model inputs
            //      if matched, check if subgraph is fully enclosed
            //      insert new merged layer
            for (int l = 0; l < model.layers.Count; ++l)
            {
                Layer layer = model.layers[l];

                foreach (var item in remappingRules)
                {
                    layersInPattern.Clear();
                    inputLayers.Clear();
                    inputConstants.Clear();

                    var pattern = item.Key(x);
                    if (!Validate(pattern, layer))
                        continue;

                    if (!CheckGraphFullInclusion(layer, layersInPattern, out string input))
                        continue;

                    var remapping = item.Value;
                    var remapLayer = remapping(layer, inputLayers, inputConstants);

                    bool unconnectedOutputs = false;
                    foreach (var layerToDelete in layersInPattern)
                    {
                        unconnectedOutputs |= (remapLayer.name != layerToDelete.name) && outputs.Contains(layerToDelete.name);
                    }
                    if (unconnectedOutputs)
                        break;


                    model.layers[nameToIndex[remapLayer.name]] = remapLayer;

                    foreach (var layerToDelete in layersInPattern)
                    {
                        if (remapLayer.name != layerToDelete.name)
                            layersToRemove.Add(layerToDelete.name);
                    }

                    break;
                }
            }

            model.layers.RemoveAll(l => layersToRemove.Contains(l.name));
        }
    }
}
