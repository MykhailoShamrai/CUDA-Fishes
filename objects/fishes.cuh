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
	void h_CleanMemoryForFishes();
	void d_CleanMemoryForFishes();
public:
	Fishes(int n, bool onGpu);
	~Fishes();

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

	
	enum FishType
	{
		NormalFish,
		LeaderOfNormalFishes,
		Predator
	};

	void GenerateRandomFishes(int width, int height, float minVel, float maxVel);
	void GenerateTestFishes();
};

struct cellForFishFunctor
{
private:
	float* xPosition;
	float* yPosition;
	float sizeOfCell;
	float width;
	float height;
public:
	cellForFishFunctor(float* x_pos, float* y_pos, float sizeOfCell, float widht, float height) :
		xPosition(x_pos), yPosition(y_pos), sizeOfCell(sizeOfCell), width(width), height(height) {};

	__host__ __device__ int operator()(int& index)
	{
		float x = xPosition[index];
		float y = yPosition[index];
		float transformed_x = x + width / 2;
		float transformed_y = y + height / 2;
		int x_index = (transformed_x + transformed_x - 1) / sizeOfCell;
		int y_index = (transformed_y + transformed_y - 1) / sizeOfCell;
		return x_index * y_index;
	}
};