#define _USE_MATH_DEFINES

#include "fishes.cuh"
#include <stdlib.h>
#include "../third_party/cuda-samples/helper_math.h"
#include "grid.cuh"
#include "options.cuh"
#include "../include/helpers.cuh"
#include <cmath>

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

float rand_float_test(float low, float high)
{
	return low + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (high - low)));
}

void Fishes::GenerateRandomFishes(int width, int height, float minVel, float maxVel)
{
	int highWidth = float(width) / 2;
	int lowWidht = -highWidth;
	int highHeight = float(height) / 2;
	int lowHeight = -highHeight;
	for (int i = 0; i < this->n; i++)
	{
		this->x_before_movement[i] = rand_float_test(lowWidht, highWidth);
		this->y_before_movement[i] = rand_float_test(lowHeight, highHeight);
		// Random normal vector in 2D
		float2 vel = float2();
		vel.x = 2.0f;
		vel.y = 2.0f;
		//float velValue = rand_float(minVel, maxVel);
		//vel *= velValue;
		this->x_vel_before_movement[i] = vel.x;
		this->y_vel_before_movement[i] = vel.y;
		// TODO: At this moment hardcoded NormalFishes
		this->types[i] = FishType::NormalFish;
	}
}

void Fishes::GenerateTestFishes()
{
	// I'll generate test 20 fishes with same velocity 1 and same vectors of velocity
	for (int i = 0; i < this->n; i++)
	{
		this->x_before_movement[i] = -100 + i * 10 - 1;
		this->y_before_movement[i] = -100 + i * 10 - 1;
		this->x_vel_before_movement[i] = 1 + 0.5f;
		this->y_vel_before_movement[i] = 1 + sqrtf(0.75);
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

__host__ __device__ bool CheckIfCellsAreNotEqual(int* table, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = i + 1; j < n; j++)
		{
			if (table[i] == table[j])
				return false;
		}
	}
	return true;
}

__host__ __device__ bool CheckCell(int cellIndex, int start, int i)
{
	if (start < 0 && i == 0)
	{
		printf("Error in: One fish at least does exist %d\n", cellIndex);
		return false;
	}
	return true;
}

__host__ __device__ int Fishes::CountForAFish(int index, Grid* grid, Options* options)
{
	float maxVel = options->maxVelNormalFishes;
	float minVel = options->minVelNormalFishes;
	float cohesionNormal = options->cohesionNormalFishes;
	float alignmentNormal = options->alignmentNormalFishes;
	float separationNormal = options->separationNormalFishes;
	int width = options->width;
	int height = options->height;

	int indexOfFish = grid->fish_id[index];

	int indexOfCell = grid->cell_id[index];

	int numberOfCells = grid->ReturnNumberOfCells();
	int numberOfCells_x = grid->ReturnNumberOfCellsX();
	int numberOfCells_y = grid->ReturnNumberOfCellsY();

	int x_ind = indexOfCell % numberOfCells_x;
	int y_ind = indexOfCell / numberOfCells_x;
	// Four cells for each quarter
	int cellsForSearch[4];
	int quarterNumber = grid->quarter_number[index];
	cellsForSearch[0] = indexOfCell;
	int x1;
	int x2;
	int x3;
	int y1;
	int y2;
	int y3;
	int x_rr = (x_ind + 1) % numberOfCells_x;
	int x_ll = (x_ind - 1) >= 0 ? x_ind - 1 : numberOfCells_x - 1;
	int y_tt = (y_ind - 1) >= 0 ? y_ind - 1 : numberOfCells_y - 1;
	int y_bb = (y_ind + 1) % numberOfCells_y;
	// Finding where should we check fishes for interaction
	switch (quarterNumber)
	{
	case 1:
		x1 = x_rr;
		y1 = y_ind;
		x2 = x_rr;
		y2 = y_tt;
		x3 = x_ind;
		y3 = y_tt;
		break;
	case 2:
		x1 = x_ind;
		y1 = y_tt;
		x2 = x_ll;
		y2 = y_tt;
		x3 = x_ll;
		y3 = y_ind;
		break;
	case 3:
		x1 = x_ll;
		y1 = y_ind;
		x2 = x_ll;
		y2 = y_bb;
		x3 = x_ind;
		y3 = y_bb;
		break;
	case 4:
		x1 = x_ind;
		y1 = y_bb;
		x2 = x_rr;
		y2 = y_bb;
		x3 = x_rr;
		y3 = y_ind;
		break;
	default:
		break;
	}
	cellsForSearch[1] = x1 + numberOfCells_x * y1;
	cellsForSearch[2] = x2 + numberOfCells_x * y2;
	cellsForSearch[3] = x3 + numberOfCells_x * y3;

	assert(CheckIfCellsAreNotEqual(cellsForSearch, 4));
	// Interaction counting 
	float2 fishPosition = float2();
	fishPosition.x = x_before_movement[indexOfFish];
	fishPosition.y = y_before_movement[indexOfFish];

	float2 alignmentPart = float2();
	alignmentPart.x = 0.0f;
	alignmentPart.y = 0.0f;
	int numberOfFriends = 0;
	for (int i = 0; i < 4; i++)
	{
		int cellStart = grid->cells_starts[cellsForSearch[i]];
		int cellEnd = grid->cells_ends[cellsForSearch[i]];
		assert(CheckCell(cellsForSearch[i], cellStart, i));
		assert(CheckCell(cellsForSearch[i], cellEnd, i));

		if (cellStart >= 0 && cellEnd >= 0)
		{
			for (int j = cellStart; j <= cellEnd; j++)
			{
				// If it's not the same fish
				int friendId = grid->fish_id[j];
				if (friendId != indexOfFish)
				{
					float2 posOfFriend;
					posOfFriend.x = x_before_movement[friendId];
					posOfFriend.y = y_before_movement[friendId];
				}
			}
		}
	}


	float2 velBeforeInteraction = float2();
	velBeforeInteraction.x = x_vel_before_movement[indexOfFish];
	velBeforeInteraction.y = y_vel_before_movement[indexOfFish];
	float2 additionalVel = float2();
	additionalVel.x = 0.0f;
	additionalVel.y = 0.0f;
	float2 velAfterCount = velBeforeInteraction + additionalVel;
	float valueOfVel = cuda_examples::length(velAfterCount);
	float2 directionVect = cuda_examples::normalize(velAfterCount);
	if (valueOfVel > maxVel)
	{
		velAfterCount = directionVect * maxVel;
	}
	else if (valueOfVel < minVel)
	{
		velAfterCount = directionVect * minVel;
	}
	// Adding velocity to position and also adding changing velocity in an array
	float xAfterMovement = x_before_movement[indexOfFish] + velAfterCount.x;
	float yAfterMovement = y_before_movement[indexOfFish] + velAfterCount.y;
	int widthHalf = width / 2;
	int heightHalf = height / 2;
	xAfterMovement = xAfterMovement > widthHalf ? -width + xAfterMovement : xAfterMovement;
	xAfterMovement = xAfterMovement < -widthHalf ? width + xAfterMovement : xAfterMovement;

	yAfterMovement = yAfterMovement > heightHalf ? -height + yAfterMovement : yAfterMovement;
	yAfterMovement = yAfterMovement < -heightHalf ? height + yAfterMovement : yAfterMovement;

	x_after_movement[indexOfFish] = xAfterMovement;
	y_after_movement[indexOfFish] = yAfterMovement;
	x_vel_after_movement[indexOfFish] = velAfterCount.x;
	y_vel_after_movement[indexOfFish] = velAfterCount.y;

	return indexOfFish;
}

__host__ __device__ void Fishes::FindTrianglesForAFish(int index, float* buffer, int lenOfTriang, int widthOfTriang)
{
	float2 currentPosition;
	currentPosition.x = x_after_movement[index];
	currentPosition.y = y_after_movement[index];
	float2 vel = float2();
	vel.x = x_vel_after_movement[index];
	vel.y = y_vel_after_movement[index];
	float2 direction = cuda_examples::normalize(vel);
	float2 reversedDirection = -direction;
	float2 normal = float2();
	normal.x = -direction.y;
	normal.y = direction.x;
	float2 first = 4 * direction + currentPosition;
	float2 second = 2 * normal + currentPosition;
	float2 third = 2 * -normal + currentPosition;
	int indexInBuffer = index * 6; // W have 6 elements for each fish
	buffer[indexInBuffer] = first.x;
	buffer[indexInBuffer + 1] = first.y;
	buffer[indexInBuffer + 2] = second.x;
	buffer[indexInBuffer + 3] = second.y;
	buffer[indexInBuffer + 4] = third.x;
	buffer[indexInBuffer + 5] = third.y;
}

__host__ __device__ void Fishes::FindCircleForFish(int index, float* buffer, int radius, int number_of_points)
{
	float2 currentPosition;
	currentPosition.x = x_after_movement[index];
	currentPosition.y = y_after_movement[index];
	int indexInBuffer = number_of_points * 2 * index;
	float2 directVect;
	directVect.x = 1.0f;
	directVect.y = 0.0f;
	float step = 2.0f * M_PI / number_of_points;
	for (int i = 0; i < number_of_points; i++)
	{
		float step_now = step * i;
		float2 vect = float2();
		vect.x = directVect.x * cosf(step_now) - directVect.y * sinf(step_now);
		vect.y = directVect.x * sinf(step_now) + directVect.y * sinf(step_now);
		buffer[indexInBuffer++] = currentPosition.x + vect.x * radius;
		buffer[indexInBuffer++] = currentPosition.y + vect.y * radius;
	}
}






