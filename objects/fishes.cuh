#pragma once
#include <cuda_runtime.h>
#include <cassert>


struct Options;
struct Grid;

struct Fishes
{
private:
	bool onGpu;
	int n;
	// Radius of visibility
	void h_AllocateMemoryForFishes();
	void d_AllocateMemoryForFishes();
public:
	Fishes(int n, bool onGpu);
	~Fishes() {};

	float* x_before_movement;
	float* y_before_movement;
	float* x_vel_before_movement;
	float* y_vel_before_movement;
	 
	float* x_after_movement;
	float* y_after_movement;
	float* x_vel_after_movement;
	float* y_vel_after_movement;

	void h_CleanMemoryForFishes();
	void d_CleanMemoryForFishes();

	void GenerateRandomFishes(int width, int height, float minVel, float maxVel);
	void GenerateTestFishes();

	void d_CopyFishesFromCPU(Fishes& fishes);
	void h_CopyFishesFromGPU(Fishes& fishes);

	__host__ __device__ int CountForAFish(int index, Grid* grid, Options* options, float cursorX, float cursorY, bool fearingWithCursor);
	__host__ __device__ void FindTrianglesForAFish(int index, float* buffer);
	__host__ __device__ void FindCircleForFish(int index, float* buffer, int radius, int number_of_points);
};


