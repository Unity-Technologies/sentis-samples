using System;
using UnityEngine;

namespace Unity.Sentis.Layers
{
    /// <summary>
    /// Represents the fused multihead attention layer based on the PyTorch implementation. This layer computes attention outputs given input tensors.
    /// </summary>
    [Serializable]
    public class SingleHeadAttention : Layer
    {
        /// <summary>
        /// The scaling factor for scaling the query.
        /// </summary>
        public float scaling_factor;

        /// <summary>
        /// Creates a fused `Multihead Attention` layer based on the PyTorch implementation.
        /// </summary>
        /// <param name="name">The name to use for the output tensor of the layer.</param>
        /// <param name="query">The name to use for the query tensor of the layer.</param>
        /// <param name="q_weight">The name to use for the weight tensor of query.</param>
        /// <param name="q_bias">The name to use for the bias tensor of the query.</param>
        /// <param name="k_weight">The name to use for the weight tensor of key.</param>
        /// <param name="k_bias">The name to use for the bias tensor of key.</param>
        /// <param name="v_weight">The name to use for the weight tensor of value.</param>
        /// <param name="v_bias">The name to use for the bias tensor of value.</param>
        /// <param name="out_weight">The name to use for the weight tensor of the out-projection layer.</param>
        /// <param name="out_bias">The name to use for the bias tensor of the out-projection layer.</param>
        /// <param name="scaling_factor">The scaling factor for scaling the query.</param>
        public SingleHeadAttention(string name, string query, string q_weight, string q_bias, string k_weight, string k_bias, string v_weight, string v_bias, string out_weight, string out_bias, float scaling_factor)
        {
            this.name = name;
            inputs = new[] { query, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias, out_weight, out_bias };
            this.scaling_factor = scaling_factor;
        }

        /// <inheritdoc/>
        internal override PartialTensor InferPartialTensor(PartialTensor[] inputTensors, PartialInferenceContext ctx)
        {
            var dataType = inputTensors[0].dataType;
            var shapeQuery = inputTensors[0].shape;

            //// Ensure that the query, key, and value tensors have compatible shapes for attention.
            //Logger.AssertIsTrue(shapeQuery.rank == shapeKey.rank && shapeKey.rank == shapeValue.rank, "RankError: ");
            //
            //// Additional checks for compatibility of dimensions
            //Logger.AssertIsTrue(shapeKey == shapeValue, "Shape of key, value must match");
            //Logger.AssertIsTrue(shapeQuery[-1] == shapeKey[-1], "The feature dim of query, key, value must be equal.");

            var shapeOut = new SymbolicTensorShape(shapeQuery[0] * shapeQuery[1], shapeQuery[2]);

            return new PartialTensor(dataType, shapeOut);
        }

        /// <inheritdoc/>
        public override Tensor Execute(Tensor[] inputTensors, ExecutionContext ctx)
        {
            var x = inputTensors[0] as TensorFloat;
            var O = ctx.backend.NewOutputTensorFloat(x.shape);
            if (O.shape.HasZeroDims())
                return O;

            var w_q = inputTensors[1] as TensorFloat;
            var b_q = inputTensors[2] as TensorFloat;

            var w_k = inputTensors[3] as TensorFloat;
            var b_k = inputTensors[4] as TensorFloat;

            var w_v = inputTensors[5] as TensorFloat;
            var b_v = inputTensors[6] as TensorFloat;

            var w_o = inputTensors[7] as TensorFloat;
            var b_o = inputTensors[8] as TensorFloat;

            var q = ctx.backend.NewTempTensorFloat(x.shape);
            ctx.backend.Dense(x, w_q, b_q, q, FusableActivation.None);
            var k = ctx.backend.NewTempTensorFloat(x.shape);
            ctx.backend.Dense(x, w_k, b_k, k, FusableActivation.None);
            var v = ctx.backend.NewTempTensorFloat(x.shape);
            ctx.backend.Dense(x, w_v, b_v, v, FusableActivation.None);

            var batch = x.shape[0];
            var sequence_length = x.shape[1];
            var embed_dim = x.shape[-1];
            var num_heads = (int)Math.Round(embed_dim * scaling_factor * scaling_factor);
            var head_dim = embed_dim / num_heads;

            var q_reshape = ctx.backend.NewTempTensorFloat(new TensorShape(batch, sequence_length, num_heads, head_dim));
            ctx.backend.Reshape(q, q_reshape);
            var k_reshape = ctx.backend.NewTempTensorFloat(new TensorShape(batch, sequence_length, num_heads, head_dim));
            ctx.backend.Reshape(k, k_reshape);
            var v_reshape = ctx.backend.NewTempTensorFloat(new TensorShape(batch, sequence_length, num_heads, head_dim));
            ctx.backend.Reshape(v, v_reshape);
            var q_transpose = ctx.backend.NewTempTensorFloat(q_reshape.shape.Transpose(new[] { 0, 2, 1, 3 }));
            ctx.backend.Transpose(q_reshape, q_transpose, new[] { 0, 2, 1, 3 });
            var k_transpose = ctx.backend.NewTempTensorFloat(k_reshape.shape.Transpose(new[] { 0, 2, 3, 1 }));
            ctx.backend.Transpose(k_reshape, k_transpose, new[] { 0, 2, 3, 1 });
            var v_transpose = ctx.backend.NewTempTensorFloat(v_reshape.shape.Transpose(new[] { 0, 2, 1, 3 }));
            ctx.backend.Transpose(v_reshape, v_transpose, new[] { 0, 2, 1, 3 });
            var qk = ctx.backend.NewTempTensorFloat(q_transpose.shape.MatMul(k_transpose.shape));
            ctx.backend.MatMul(q_transpose, k_transpose, qk);
            var qks = ctx.backend.NewTempTensorFloat(qk.shape);
            ctx.backend.ScalarMad(qk, qks, scaling_factor, 0.0f);
            var attn_output_weights = ctx.backend.NewTempTensorFloat(qks.shape);
            ctx.backend.Softmax(qks, attn_output_weights, -1);
            var attn_output = ctx.backend.NewTempTensorFloat(attn_output_weights.shape.MatMul(v_transpose.shape));
            ctx.backend.MatMul(attn_output_weights, v_transpose, attn_output);
            var attn_output_transpose = ctx.backend.NewTempTensorFloat(attn_output.shape.Transpose(new[] { 0, 2, 1, 3 }));
            ctx.backend.Transpose(attn_output, attn_output_transpose, new[] { 0, 2, 1, 3 });
            var attn_output_reshape = ctx.backend.NewTempTensorFloat(new TensorShape(batch, sequence_length, embed_dim));
            ctx.backend.Reshape(attn_output_transpose, attn_output_reshape);
            ctx.backend.Dense(attn_output_reshape, w_o, b_o, O, FusableActivation.None);
            return O;
        }

        internal override string profilerTag => "SingleHeadAttention";
    }

}
