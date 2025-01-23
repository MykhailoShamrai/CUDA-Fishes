#include "main_loop_gpu.cuh"
#include "../objects/fishes.cuh"
#include "../objects/grid.cuh"
#include "../objects/options.cuh"
#include <device_launch_parameters.h>
#include <stdio.h>
#include <assert.h>

__host__ __device__ static bool VerifyIndex(int index, int max_n)
{
	return index < max_n;
}
__global__ void CountForFishesGpu(Grid grid, Options* options, Fishes fishes, float* buffer, int n)
{
	__shared__ Options shared_options;
	if (threadIdx.x == 0)
	{
		shared_options = *options;
	}
	__syncthreads();

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n)
	{
		return;
	}
	assert(VerifyIndex(i, n));
	int indexOfFish = fishes.CountForAFish(i, &grid, options);
	// Hardcoded parameters for triangles
	fishes.FindTrianglesForAFish(indexOfFish, buffer);
}

__global__ void CountCircleForFishesGpu(Fishes fishes, float* buffer, int n_fishes, int n_points, int radius)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n_fishes)
	{
		return;
	}
	assert(VerifyIndex(i, n_fishes));
	fishes.FindCircleForFish(i, buffer, radius, n_points);
}