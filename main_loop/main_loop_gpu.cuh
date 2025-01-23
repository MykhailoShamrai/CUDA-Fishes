#pragma once

#include <cuda_runtime.h>
struct Fishes;
struct Grid;
struct Options;

__global__ void CountForFishesGpu(Grid grid, Options* options, Fishes fishes,
	float* buffer, int n, float cursorPosX, float cursorPosY, bool fearingWithCursor);

__global__ void CountCircleForFishesGpu(Fishes fishes, float* buffer, int n_fishes,
	int n_points, int radius);
