#include "main_loop_gpu.cuh"
#include "../objects/fishes.cuh"
#include "../objects/grid.cuh"
#include "../objects/options.cuh"
#include <device_launch_parameters.h>
#include <stdio.h>

__global__ void CountForFishes(Grid grid, Options* options, Fishes fishes, float* buffer, int n)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n)
		return;
	int indexOfFish = fishes.CountForAFish(i, &grid, options);
	// Hardcoded parameters for triangles
	fishes.FindTrianglesForAFish(indexOfFish, buffer, 10, 6);
}