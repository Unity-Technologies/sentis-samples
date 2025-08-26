using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace Unity.ML.Tokenization.Tokenizers
{
    partial class BpeTokenizer
    {
        internal class Merger : IManyToManyConverter<ITokenDefinition, ITokenDefinition>
        {
            readonly Pool<PriorityQueue<Mergeable>> m_MergeableQueuePool = new(
                () => new(Compare), l => l.Clear());

            readonly ReadOnlyDictionary<long, (int rank, ITokenDefinition definition)> m_Merges;

            public Merger(
                IEnumerable<(ITokenDefinition a, ITokenDefinition b, ITokenDefinition merged, int
                    rank)> merges)
            {
                var mergesLut = new Dictionary<long, (int rank, ITokenDefinition definition)>();
                foreach (var (a, b, merged, rank) in merges)
                    mergesLut.Add(GetPairId(a.Id, b.Id), (rank, merged));

                m_Merges = new ReadOnlyDictionary<long, (int, ITokenDefinition)>(mergesLut);
            }

            public void Convert(
                IEnumerable<ITokenDefinition> input,
                IOutput<ITokenDefinition> output)
            {
                var symbols = input
                    .Select(
                        (definition, position) => new Symbol
                        {
                            Definition = definition,
                            Position = position,
                            Previous = position - 1,
                            Next = position + 1,
                            Discarded = false
                        })
                    .ToArray();

                symbols[^1].Next = -1;

                Merge(symbols);

                output.Add(symbols.Where(s => !s.Discarded).Select(s => s.Definition));
            }

            static IEnumerable<(T a, T b)> GetPairs<T>(IEnumerable<T> input)
            {
                using var inputEnum = input.GetEnumerator();

                if (!inputEnum.MoveNext())
                    yield break;

                var a = inputEnum.Current;

                while (inputEnum.MoveNext())
                {
                    var b = inputEnum.Current;
                    yield return (a, b);
                    a = b;
                }
            }

            static int Compare(Mergeable a, Mergeable b)
            {
                var rank = a.rank - b.rank;
                return rank != 0 ? rank : a.position - b.position;
            }


            static long GetPairId(int a, int b) => ((long) a << 32) | (uint) b;

            void Merge(Symbol[] symbols)
            {
                using var _ = m_MergeableQueuePool.Get(out var mergeableQueue);

                foreach (var (a, b) in GetPairs(symbols.Select(s => s.Position)))
                    TryAddMergeable(ref symbols[a], ref symbols[b], mergeableQueue);

                while (mergeableQueue.TryPop(out var mergeable))
                {
                    ref var symbolA = ref symbols[mergeable.position];
                    if (symbolA.Discarded || symbolA.Next == -1)
                        continue;

                    ref var symbolB = ref symbols[symbolA.Next];

                    if (symbolB.Discarded)
                        continue;

                    var pairId = GetPairId(symbolA.Definition.Id, symbolB.Definition.Id);
                    if (!m_Merges.TryGetValue(pairId, out var merged))
                        continue;

                    if (!ReferenceEquals(merged.definition, mergeable.definition))
                        continue;

                    symbolA.Definition = merged.definition;
                    symbolB.Discarded = true;

                    symbolA.Next = symbolB.Next;
                    if (symbolB.Next != -1)
                        symbols[symbolB.Next].Previous = symbolA.Position;

                    // Try pair with previous
                    if (symbolA.Previous != -1)
                        TryAddMergeable(ref symbols[symbolA.Previous], ref symbolA, mergeableQueue);

                    // Try pair with next
                    if (symbolA.Next != -1)
                        TryAddMergeable(ref symbolA, ref symbols[symbolA.Next], mergeableQueue);
                }
            }

            void TryAddMergeable(ref Symbol a, ref Symbol b, PriorityQueue<Mergeable> target)
            {
                var pairId = GetPairId(a.Definition.Id, b.Definition.Id);

                if (!m_Merges.TryGetValue(pairId, out var merged))
                    return;

                var mergeable = new Mergeable
                {
                    definition = merged.definition, position = a.Position, rank = merged.rank
                };

                target.Push(mergeable);
            }
        }
    }
}
