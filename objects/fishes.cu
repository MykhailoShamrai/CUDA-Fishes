#include "fishes.cuh"
#include <stdlib.h>
#include "../include/helpers.cuh"
#include "../third_party/cuda-samples/helper_math.h"


Fishes::Fishes(int n, bool onGpu): n(n), onGpu(onGpu)
{
	if (onGpu)
	{
		d_AllocateMemoryForFishes();
	}
	else
	{
		h_AllocateMemoryForFishes();
	}
}


void Fishes::h_AllocateMemoryForFishes()
{
	this->x_before_movement = (float*)malloc(sizeof(float) * n);
	this->y_before_movement = (float*)malloc(sizeof(float) * n);
	this->x_vel_before_movement = (float*)malloc(sizeof(float) * n);
	this->y_vel_before_movement = (float*)malloc(sizeof(float) * n);

	this->x_after_movement = (float*)malloc(sizeof(float) * n);
	this->y_after_movement = (float*)malloc(sizeof(float) * n);
	this->x_vel_after_movement = (float*)malloc(sizeof(float) * n);
	this->y_vel_after_movement = (float*)malloc(sizeof(float) * n);

	this->types = (Fishes::FishType*)malloc(sizeof(Fishes::FishType) * n);
} 

void Fishes::d_AllocateMemoryForFishes()
{
	checkCudaErrors(cudaMalloc((void**)&this->x_before_movement, sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&this->y_before_movement, sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&this->x_vel_before_movement, sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&this->y_vel_before_movement, sizeof(float) * n));

	checkCudaErrors(cudaMalloc((void**)&this->x_after_movement, sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&this->y_after_movement, sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&this->x_vel_after_movement, sizeof(float) * n));
	checkCudaErrors(cudaMalloc((void**)&this->y_vel_after_movement, sizeof(float) * n));

	checkCudaErrors(cudaMalloc((void**)&this->types, sizeof(Fishes::FishType) * n));
}

void Fishes::h_CleanMemoryForFishes()
{
	free(this->x_before_movement);
	free(this->y_before_movement);
	free(this->x_vel_before_movement);
	free(this->y_vel_before_movement);

	free(this->x_after_movement);
	free(this->y_after_movement);
	free(this->x_vel_after_movement);
	free(this->y_vel_after_movement);

	free(this->types);
}

void Fishes::d_CleanMemoryForFishes()
{
	checkCudaErrors(cudaFree(x_before_movement));
	checkCudaErrors(cudaFree(y_before_movement));
	checkCudaErrors(cudaFree(x_vel_before_movement));
	checkCudaErrors(cudaFree(y_vel_before_movement));

	checkCudaErrors(cudaFree(x_after_movement));
	checkCudaErrors(cudaFree(y_after_movement));
	checkCudaErrors(cudaFree(x_vel_after_movement));
	checkCudaErrors(cudaFree(y_vel_after_movement));

	checkCudaErrors(cudaFree(types));
}

void Fishes::GenerateRandomFishes(int width, int height, float minVel, float maxVel)
{
	int highWidth = float(width) / 2;
	int lowWidht = -highWidth;
	int highHeight = float(height) / 2;
	int lowHeight = float(height) / 2;
	for (int i = 0; i < this->n; i++)
	{
		this->x_before_movement[i] = rand_float(lowWidht, highWidth);
		this->y_before_movement[i] = rand_float(lowHeight, highHeight);
		// Random normal vector in 2D
		float2 vel = float2();
		vel.x = rand_float(-1.0f, 1.0f);
		vel.y = sqrtf(1.0f - vel.x * vel.x);
		if (rand_float(0.0f, 1.0f) < 0.5f)
		{
			vel.y = -vel.y;
		}
		float velValue = rand_float(minVel, maxVel);
		vel *= velValue;
		this->x_vel_before_movement[i] = vel.x;
		this->y_vel_before_movement[i] = vel.y;
		// TODO: At this moment hardcoded NormalFishes
		this->types[i] = FishType::NormalFish;
	}
}

void Fishes::GenerateTestFishes()
{
	// I'll generate test 20 fishes with same velocity 1 and same vectors of velocity
	for (int i = 0; i < 20; i++)
	{
		this->x_before_movement[i] = -100 + i * 10 - 1;
		this->y_before_movement[i] = -100 + i * 10 - 1;
		this->x_vel_before_movement[i] = 10 * 0.5f;
		this->y_vel_before_movement[i] = 10 * sqrtf(0.75);
		this->types[i] = FishType::NormalFish;
	}
}

void Fishes::d_CopyFishesFromCPU(float* x_before_movement, float* y_before_movement, float* x_vel_before_movement,
	float* y_vel_before_movement, FishType* types)
{
	if (onGpu)
	{
		checkCudaErrors(cudaMemcpy(this->x_before_movement, x_before_movement, n * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->y_before_movement, y_before_movement, n * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->x_vel_before_movement, x_vel_before_movement, n * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->y_vel_before_movement, y_vel_before_movement, n * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->types, types, n * sizeof(FishType), cudaMemcpyHostToDevice));
	}
}

