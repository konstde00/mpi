using System;
using System.Diagnostics;
using System.Linq;
using MPI;

class Program
{
    static void Main(string[] args)
    {
        using (new MPI.Environment(ref args))
        {
            var comm = Communicator.world;
            int rank = comm.Rank;
            int size = comm.Size;

            int[] matrixSizes = { 1000, 2000, 5000 };
            double[,] timings = new double[matrixSizes.Length, 4];

            foreach (int testIndex in Enumerable.Range(0, matrixSizes.Length))
            {
                int N = matrixSizes[testIndex]; // N x N матриця
                if (rank == 0) Console.WriteLine($"\nРозмiр: матрицi {N}x{N}");

                // Підрахунок рядків для кожного процесу
                int baseRows = N / size;
                int extra = N % size;
                int myRows = baseRows + (rank < extra ? 1 : 0);
                int[] rowCounts = Enumerable.Range(0, size)
                    .Select(r => baseRows + (r < extra ? 1 : 0)).ToArray();

                // Генерація матриць (тільки на root)
                double[] flatA = null, flatB = null;
                if (rank == 0)
                {
                    Random rnd = new Random(42);
                    flatA = Enumerable.Range(0, N * N).Select(_ => rnd.NextDouble()).ToArray();
                    flatB = Enumerable.Range(0, N * N).Select(_ => rnd.NextDouble()).ToArray();
                }

                // Підготовка блоків для Scatter
                double[][] blocksA = null, blocksB = null;
                if (rank == 0)
                {
                    blocksA = SplitMatrix(flatA, N, rowCounts);
                    blocksB = SplitMatrix(flatB, N, rowCounts);
                }

                double[] localA = comm.Scatter(blocksA, 0);
                double[] localB = comm.Scatter(blocksB, 0);
                int localSize = localA.Length;

                for (int op = 0; op < 4; op++)
                {
                    double[] localResult = new double[localSize];
                    Stopwatch sw = null;
                    if (rank == 0) sw = Stopwatch.StartNew();

                    switch (op)
                    {
                        case 0: // Додавання
                            for (int i = 0; i < localSize; i++)
                                localResult[i] = localA[i] + localB[i];
                            break;
                        case 1: // Віднімання
                            for (int i = 0; i < localSize; i++)
                                localResult[i] = localA[i] - localB[i];
                            break;
                        case 2: // Множення на скаляр
                            double scalar = 2.5;
                            for (int i = 0; i < localSize; i++)
                                localResult[i] = localA[i] * scalar;
                            break;
                        case 3: // Транспонування (локальне)
                            int rows = rowCounts[rank];
                            double[] localTransposed = new double[localSize];
                            for (int i = 0; i < rows; i++)
                                for (int j = 0; j < N; j++)
                                    localTransposed[j * rows + i] = localA[i * N + j];
                            localResult = localTransposed;
                            break;
                    }

                    double[][] gathered = comm.Gather(localResult, 0);

                    if (rank == 0)
                    {
                        sw.Stop();
                        double[] resultMatrix = MergeBlocks(gathered);
                        timings[testIndex, op] = sw.Elapsed.TotalSeconds;
                    }
                }
            }

            if (rank == 0)
            {
                Console.WriteLine("\nРозмiр\tДодавання\tВiднiмання\tМнож. на скаляр\tТранспонування");
                for (int i = 0; i < matrixSizes.Length; i++)
                {
                    Console.WriteLine($"{matrixSizes[i]}x{matrixSizes[i]}\t" +
                        $"{timings[i, 0]:0.0000}\t\t{timings[i, 1]:0.0000}\t\t{timings[i, 2]:0.0000}\t\t{timings[i, 3]:0.0000}");
                }
                Console.ReadKey();
            }
        }
    }

    static double[][] SplitMatrix(double[] flatMatrix, int N, int[] rowCounts)
    {
        double[][] result = new double[rowCounts.Length][];
        int offset = 0;
        for (int i = 0; i < rowCounts.Length; i++)
        {
            int rows = rowCounts[i];
            int count = rows * N;
            result[i] = new double[count];
            Array.Copy(flatMatrix, offset, result[i], 0, count);
            offset += count;
        }
        return result;
    }

    static double[] MergeBlocks(double[][] blocks)
    {
        int totalSize = blocks.Sum(b => b.Length);
        double[] merged = new double[totalSize];
        int offset = 0;
        foreach (var block in blocks)
        {
            Array.Copy(block, 0, merged, offset, block.Length);
            offset += block.Length;
        }
        return merged;
    }
}
