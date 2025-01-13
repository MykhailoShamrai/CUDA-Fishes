#include "fishes.cuh"
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../include/helpers.cuh"


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

Fishes::~Fishes()
{
	if (onGpu)
	{
		d_CleanMemoryForFishes();
	}
	else
	{
		h_CleanMemoryForFishes();
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
	checkCudaErrors(cudaFree(this->x_before_movement));
	checkCudaErrors(cudaFree(this->y_before_movement));
	checkCudaErrors(cudaFree(this->x_vel_before_movement));
	checkCudaErrors(cudaFree(this->y_vel_before_movement));

	checkCudaErrors(cudaFree(this->x_after_movement));
	checkCudaErrors(cudaFree(this->y_after_movement));
	checkCudaErrors(cudaFree(this->x_vel_after_movement));
	checkCudaErrors(cudaFree(this->y_vel_after_movement));

	checkCudaErrors(cudaFree(this->types));
}