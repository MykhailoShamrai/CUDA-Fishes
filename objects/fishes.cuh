#pragma once
#include <cuda_runtime.h>
#include <cassert>


extern struct Options;
extern struct Grid;

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

	__host__ __device__ int CountForAFish(int index, Grid* grid, Options* options);
	__host__ __device__ void FindTrianglesForAFish(int index, float* buffer, int lenOfTriang, int widthOfTriang);
};

struct CopyFishPositionsAndVelocitiesAfterCountFunctor
{
private:
	float* x_before_movement;
	float* y_before_movement;
	float* x_vel_before_movement;
	float* y_vel_before_movement;

	float* x_after_movement;
	float* y_after_movement;
	float* x_vel_after_movement;
	float* y_vel_after_movement;
public:
	CopyFishPositionsAndVelocitiesAfterCountFunctor(float* xBefore, float* yBefore, float* xVelBefore,
		float* yVelBefore, float* xAfter, float* yAfter, float* xVelAfter, float* yVelAfter) :
		x_before_movement(xBefore), y_before_movement(yBefore), x_vel_before_movement(xVelBefore),
		y_vel_before_movement(yVelBefore), x_after_movement(xAfter), y_after_movement(yAfter),
		x_vel_after_movement(xVelAfter), y_vel_after_movement(yVelAfter) {};

	__host__ __device__ int operator()(int index)
	{
		assert(x_before_movement[index] != x_after_movement[index]);
		assert(y_before_movement[index] != y_after_movement[index]);
		x_before_movement[index] = x_after_movement[index];
		y_before_movement[index] = y_after_movement[index];
		x_vel_before_movement[index] = x_vel_after_movement[index];
		y_vel_before_movement[index] = y_vel_after_movement[index];
		return index;
	}
};

