#include "main_loop_cpu.cuh"
#include "../objects/fishes.cuh"
#include "../objects/grid.cuh"
#include "../objects/options.cuh"

void CountForFishesCpu(Grid& grid, Options& options, Fishes& fishes,
	float* buffer, int n)
{
	for (int i = 0; i < n; i++)
	{
		int index = fishes.CountForAFish(i, &grid, &options);
		fishes.FindTrianglesForAFish(index, buffer);
	}
}

void CountCircleForFishesCpu(Fishes& fishes, float* buffer, int n_fishes,
	int n_points, int radius)
{
	for (int i = 0; i < n_fishes; i++)
	{
		fishes.FindCircleForFish(i, buffer, radius, n_points);
	}
}
