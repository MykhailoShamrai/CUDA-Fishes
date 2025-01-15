#pragma once
#include <cuda_runtime.h>


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

	enum FishType
	{
		NormalFish,
		LeaderOfNormalFishes,
		Predator
	};

	float* x_before_movement;
	float* y_before_movement;
	float* x_vel_before_movement;
	float* y_vel_before_movement;
	 
	float* x_after_movement;
	float* y_after_movement;
	float* x_vel_after_movement;
	float* y_vel_after_movement;

	FishType* types;


	void h_CleanMemoryForFishes();
	void d_CleanMemoryForFishes();

	void GenerateRandomFishes(int width, int height, float minVel, float maxVel);
	void GenerateTestFishes();
	void d_CopyFishesFromCPU(float* x_before_movement, float* y_before_movement,
		float* x_vel_before_movement, float* y_vel_before_movement, FishType* types);

	__host__ __device__ void CountForAFish(int index, Grid* grid, Options* options);
	__host__ __device__ void CountSeparation();
	__host__ __device__ void CountAlignment();
	__host__ __device__ void CountCohession();
};

