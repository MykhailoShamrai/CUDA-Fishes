#pragma once

#include <cuda_runtime.h>
extern struct Fishes;
extern struct Grid;
extern struct Options;

__global__ void CountForFishes(Grid grid, Options* options, Fishes fishes,
	float* buffer, int n);

__global__ void CountCircleForFish(Fishes fishes, float* buffer, int n_fishes,
	int n_points, int radius);
