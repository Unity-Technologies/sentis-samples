namespace Unity.Sentis
{
    /// <summary>
    /// Represents a tensor during PartialInference, when input data and shapes are not fully defined.
    /// </summary>
    class PartialTensor
    {
        const int k_MaxLength = TensorShape.maxRank * 2;
        DataType m_DataType;
        SymbolicTensorShape m_Shape;
        PartialTensorElement[] m_Elements;

        /// <summary>
        /// The number of partial elements stored, is equal to the length of the tensor if partial elements are stored.
        /// </summary>
        public int length => m_Elements.Length;
        /// <summary>
        /// The shape of the partial tensor as a SymbolicTensorShape.
        /// </summary>
        public SymbolicTensorShape shape => m_Shape;
        /// <summary>
        /// Whether partial elements are stored.
        /// </summary>
        public bool isPartiallyKnown => m_Elements != null;
        /// <summary>
        /// The data type of the partial tensor as a DataType.
        /// </summary>
        public DataType dataType => m_DataType;

        /// <summary>
        /// Initializes and returns a partial tensor with the specified `dataType` and 'shape'.
        /// If the shape is small enough unknown partial tensor elements are tracked and 'isPartiallyKnown' returns 'true'.
        /// </summary>
        public PartialTensor(DataType dataType, SymbolicTensorShape shape)
        {
            m_DataType = dataType;
            m_Shape = shape;
            if (shape.IsFullyKnown() && shape.Length() <= k_MaxLength)
                m_Elements = new PartialTensorElement[shape.Length().value];
        }

        /// <summary>
        /// Initializes and returns a partial tensor with the specified `dataType` and unknown shape.
        /// </summary>
        public PartialTensor(DataType dataType)
            : this(dataType, SymbolicTensorShape.UnknownShape) { }

        /// <summary>
        /// Initializes and returns a partial tensor from a given 'tensor'. The data type shape and potential partial
        /// tensor elements are inferred from the given 'tensor'.
        /// </summary>
        public static PartialTensor FromTensor(Tensor tensor)
        {
            var partialTensor = new PartialTensor(tensor.dataType, new SymbolicTensorShape(tensor.shape));
            if (!partialTensor.isPartiallyKnown)
                return partialTensor;

            if (tensor is TensorInt tensorInt)
            {
                for (var i = 0; i < partialTensor.length; i++)
                    partialTensor[i] = new PartialTensorElement(tensorInt[i]);
            }
            else if (tensor is TensorFloat tensorFloat)
            {
                for (var i = 0; i < partialTensor.length; i++)
                    partialTensor[i] = new PartialTensorElement(tensorFloat[i]);
            }

            return partialTensor;
        }

        /// <summary>
        /// Returns a partial tensor that is the most defined of two partial tensors known to represent equal tensors.
        /// e.g. if one of the input partial tensors has a fully defined dim or element then this will be used in the
        /// return partial tensor of this method.
        /// </summary>
        internal static PartialTensor MaxDefinedPartialTensor(PartialTensor a, PartialTensor b)
        {
            if (a == null)
                return b;
            if (b == null)
                return a;
            Logger.AssertIsTrue(a.dataType == b.dataType, "InputError: incompatible tensor shapes");
            if (!a.isPartiallyKnown && !b.isPartiallyKnown)
                return new PartialTensor(a.dataType, SymbolicTensorShape.MaxDefinedShape(a.shape, b.shape));
            if (!a.isPartiallyKnown)
                return b;
            if (!b.isPartiallyKnown)
                return a;
            Logger.AssertIsTrue(a.length == b.length, "InputError: incompatible tensors");
            var maxDefinedPartialTensor = new PartialTensor(a.dataType, a.shape);
            for (var i = 0; i < maxDefinedPartialTensor.length; i++)
            {
                maxDefinedPartialTensor[i] = PartialTensorElement.MaxDefinedElement(a[i], b[i]);
            }
            return maxDefinedPartialTensor;
        }

        /// <summary>
        /// Returns a new partial tensor resulting from reshaping this partial tensor with a given symbolic shape.
        /// A single unknown output dimension can be inferred when the input shape is fully known.
        /// If 'allowZeroLength' is false then this method assumes no dimensions are 0, which allows for more general
        /// inference.
        /// </summary>
        public PartialTensor Reshape(SymbolicTensorShape newShape, bool allowZeroLength = true)
        {
            if (!newShape.hasRank)
                return new PartialTensor(dataType, newShape);

            if (!newShape.IsFullyKnown())
            {
                SymbolicTensorShape.ReduceCommonFactors(shape, newShape, out var reducedPrev, out var reducedNew, !allowZeroLength);
                var reducedPrevLength = reducedPrev.Length();
                if (!reducedPrevLength.isUnknown)
                {
                    var cumValue = 1;
                    var numUnknowns = 0;
                    var unknownIndex = 0;
                    for (var i = 0; i < reducedNew.rank; i++)
                    {
                        if (reducedNew[i].isValue)
                        {
                            cumValue *= reducedNew[i].value;
                            continue;
                        }

                        numUnknowns++;
                        unknownIndex = i;
                    }

                    if (cumValue == 1 && numUnknowns == 1)
                        newShape[unknownIndex] = reducedPrevLength;
                }
            }

            var reshapedPartialTensor = new PartialTensor(dataType, newShape);
            if (!reshapedPartialTensor.isPartiallyKnown || !isPartiallyKnown)
                return reshapedPartialTensor;

            for (var i = 0; i < reshapedPartialTensor.length; i++)
                reshapedPartialTensor.m_Elements[i] = m_Elements[i];

            return reshapedPartialTensor;
        }

        /// <summary>
        /// Whether the partial tensor is fully known, i.e. the shape is known and all the partial elements are known.
        /// If so this partial tensor can be converted to a tensor.
        /// </summary>
        public bool IsFullyKnown()
        {
            if (!isPartiallyKnown)
                return false;

            for (var i = 0; i < m_Elements.Length; i++)
            {
                if (!m_Elements[i].isIntValue && !m_Elements[i].isFloatValue)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Gets or sets the element of a given index from the flattened array of partial tensor elements.
        /// </summary>
        public PartialTensorElement this[int d0]
        {
            get
            {
                if (m_Elements == null)
                    return PartialTensorElement.Unknown;
                Logger.AssertIsTrue(d0 < m_Elements.Length, "InputError: index out of bounds");
                return m_Elements[d0];
            }
            set
            {
                if (m_Elements != null)
                    m_Elements[d0] = value;
            }
        }

        public PartialTensorElement this[PartialTensorElement d0] => !d0.isIntValue ? PartialTensorElement.Unknown : this[d0.intValue];

        /// <summary>
        /// Returns a symbolic tensor shape represented by the (integer) partial tensor
        /// e.g. as an input to ConstantOfShape.
        /// </summary>
        public SymbolicTensorShape ToSymbolicTensorShape()
        {
            if (isPartiallyKnown)
            {
                var shapeOut = SymbolicTensorShape.UnknownOfRank(length);
                for (var i = 0; i < length; i++)
                {
                    shapeOut[i] = (SymbolicTensorDim)m_Elements[i];
                }

                return shapeOut;
            }

            if (!shape.hasRank)
                return SymbolicTensorShape.UnknownShape;

            Logger.AssertIsTrue(shape.rank == 1, "Shape tensor must have rank 1");

            if (shape[0].isValue)
                return SymbolicTensorShape.UnknownOfRank(shape[0].value);

            return SymbolicTensorShape.UnknownShape;
        }

        /// <summary>
        /// Returns a tensor represented by this partial tensor.
        /// If this partial tensor is not fully known returns 'null'.
        /// </summary>
        public Tensor ToTensor()
        {
            if (!IsFullyKnown())
                return null;

            if (dataType == DataType.Float)
            {
                var values = new float[length];
                for (var i = 0; i < length; i++)
                    values[i] = m_Elements[i].floatValue;
                return new TensorFloat(shape.ToTensorShape(), values);
            }
            else
            {
                var values = new int[length];
                for (var i = 0; i < length; i++)
                    values[i] = m_Elements[i].intValue;
                return new TensorInt(shape.ToTensorShape(), values);
            }
        }

        /// <summary>
        /// Creates and returns an integer partial tensor with a given 'shape' and elements all ones if the length is small enough.
        /// </summary>
        public static PartialTensor Ones(SymbolicTensorShape shape)
        {
            var partialTensor = new PartialTensor(DataType.Int, shape);
            if (!partialTensor.isPartiallyKnown)
                return partialTensor;
            for (var i = 0; i < partialTensor.length; i++)
            {
                partialTensor[i] = new PartialTensorElement(1);
            }

            return partialTensor;
        }

        /// <summary>
        /// Creates and returns an integer partial tensor with a given 'shape' and elements from an range if the length is small enough.
        /// </summary>
        public static PartialTensor Range(int start, int end)
        {
            var partialTensor = new PartialTensor(DataType.Int, new SymbolicTensorShape(new SymbolicTensorDim(end - start)));
            if (!partialTensor.isPartiallyKnown)
                return partialTensor;
            for (var i = 0; i < partialTensor.length; i++)
            {
                partialTensor[i] = new PartialTensorElement(start + i);
            }

            return partialTensor;
        }

        /// <summary>
        /// Whether two partial tensors are identical.
        /// note this can return false even when the partial tensors are compatible and represent the same tensor.
        /// </summary>
        public static bool IsEquivalent(PartialTensor a, PartialTensor b)
        {
            if (a == null || b == null)
                return false;
            if (a.dataType != b.dataType)
                return false;
            if (!a.shape.hasRank && !b.shape.hasRank)
                return true;
            if (a.shape != b.shape)
                return false;
            if (!a.isPartiallyKnown && !b.isPartiallyKnown)
                return true;
            for (var i = 0; i < a.length; i++)
            {
                if (a[i] != b[i])
                    return false;
            }

            return true;
        }
    }
}
