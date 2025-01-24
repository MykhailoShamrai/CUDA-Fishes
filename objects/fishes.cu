#define _USE_MATH_DEFINES

#include "fishes.cuh"
#include <stdlib.h>
#include "../third_party/cuda-samples/helper_math.h"
#include "grid.cuh"
#include "options.cuh"
#include "../includes/helpers.cuh"
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
		this->x_before_movement[i] = rand_float(lowWidht, highWidth);
		this->y_before_movement[i] = rand_float(lowHeight, highHeight);
		// Random normal vector in 2D
		float2 vel = float2();
		vel.x = rand_float_test(-1.0f, 1.0f);
		vel.y = rand_float_test(-1.0f, 1.0f);
		this->x_vel_before_movement[i] = vel.x;
		this->y_vel_before_movement[i] = vel.y;
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
	}
}

void Fishes::d_CopyFishesFromCPU(Fishes& fishes)
{
	if (onGpu)
	{
		checkCudaErrors(cudaMemcpy(this->x_before_movement, fishes.x_before_movement, n * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->y_before_movement, fishes.y_before_movement, n * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->x_vel_before_movement, fishes.x_vel_before_movement, n * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->y_vel_before_movement, fishes.y_vel_before_movement, n * sizeof(float), cudaMemcpyHostToDevice));

		checkCudaErrors(cudaMemcpy(this->x_after_movement, fishes.x_after_movement, n * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->y_after_movement, fishes.y_after_movement, n * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->x_vel_after_movement, fishes.x_vel_after_movement, n * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(this->y_vel_after_movement, fishes.y_vel_after_movement, n * sizeof(float), cudaMemcpyHostToDevice));
	}
}

void Fishes::h_CopyFishesFromGPU(Fishes& fishes)
{
	if (!onGpu)
	{
		checkCudaErrors(cudaMemcpy(this->x_before_movement, fishes.x_before_movement, n * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->y_before_movement, fishes.y_before_movement, n * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->x_vel_before_movement, fishes.x_vel_before_movement, n * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->y_vel_before_movement, fishes.y_vel_before_movement, n * sizeof(float), cudaMemcpyDeviceToHost));

		checkCudaErrors(cudaMemcpy(this->x_after_movement, fishes.x_after_movement, n * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->y_after_movement, fishes.y_after_movement, n * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->x_vel_after_movement, fishes.x_vel_after_movement, n * sizeof(float), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(this->y_vel_after_movement, fishes.y_vel_after_movement, n * sizeof(float), cudaMemcpyDeviceToHost));
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

__host__ __device__ void FillArrayForSearch(int indexOfCell, int quarter, int* array, int n, 
	int numberOfCellsY, int numberOfCellsX, int x_ind, int y_ind)
{
	assert(n == 4);
	array[0] = indexOfCell;
	int x1;
	int x2;
	int x3;
	int y1;
	int y2;
	int y3;
	int x_rr = (x_ind + 1) < numberOfCellsX ? x_ind + 1 : -1;
	int x_ll = (x_ind - 1) >= 0 ? x_ind - 1 : -1;
	int y_tt = (y_ind - 1) >= 0 ? y_ind - 1 : -1;
	int y_bb = (y_ind + 1) < numberOfCellsY ? y_ind + 1 : -1;
	// Finding where should we check fishes for interaction
	switch (quarter)
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
	array[1] = -1;
	array[2] = -1;
	array[3] = -1;
	if (x1 >= 0 && y1 >= 0)
	{
		array[1] = x1 + numberOfCellsX * y1;
	}
	if (x2 >= 0 && y2 >= 0)
	{
		array[2] = x2 + numberOfCellsX * y2;
	}
	if (x3 >= 0 && y3 >= 0)
	{
		array[3] = x3 + numberOfCellsX * y3;
	}
}

__host__ __device__ int Fishes::CountForAFish(int index, Grid* grid, Options* options, float cursorX, float cursorY, bool fearingWithCursor)
{
	float maxVel = options->maxVelFishes;
	float minVel = options->minVelFishes;
	float cohesionNormal = options->cohesionForFishes;
	float alignmentNormal = options->alignmentForFishes;
	float separationNormal = options->separationForFishes;
	float separationRadius = options->radiusSeparation;
	int width = options->width;
	int height = options->height;

	float radiusForFish = options->radiusForFishes;
	float angleForFish = options->angleNormalFishes;
	float wallAvoidanceKoef = options->forceForWallAvoidance;
	float rangeForWallAvoidance = options->rangeToBorderToStartTurn;
	float fearKoef = options->powerOfFearForCursor;

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
	FillArrayForSearch(indexOfCell, quarterNumber, cellsForSearch, 4, numberOfCells_y, numberOfCells_x,
		x_ind, y_ind);

	float2 fishPosition = float2();
	fishPosition.x = x_before_movement[indexOfFish];
	fishPosition.y = y_before_movement[indexOfFish];

	//assert(CheckIfCellsAreNotEqual(cellsForSearch, 4));
	// Interaction counting 
	float2 velBeforeInteraction = float2();
	velBeforeInteraction.x = x_vel_before_movement[indexOfFish];
	velBeforeInteraction.y = y_vel_before_movement[indexOfFish];

	float2 alignmentPart = float2();
	alignmentPart.x = 0.0f;
	alignmentPart.y = 0.0f;

	float2 separationPart = float2();
	separationPart.x = 0.0f;
	separationPart.y = 0.0f;

	float2 cohesionPart = float2();
	cohesionPart.x = 0.0f;
	cohesionPart.y = 0.0f;

	float2 borderAvoidance = float2();
	borderAvoidance.x = 0.0f;
	borderAvoidance.y = 0.0f;

	// Left Border
	if (fishPosition.x < -width / 2 + rangeForWallAvoidance)
	{
		borderAvoidance.x += wallAvoidanceKoef;
	}
	// Right Border
	else if (fishPosition.x > width / 2 - rangeForWallAvoidance)
	{
		borderAvoidance.x -= wallAvoidanceKoef;
	}
	// Top border
	if (fishPosition.y > height / 2 - rangeForWallAvoidance)
	{
		borderAvoidance.y -= wallAvoidanceKoef;
	}
	else if (fishPosition.y < -height / 2 + rangeForWallAvoidance)
	{
		borderAvoidance.y += wallAvoidanceKoef;
	}


	// The fish for which we are counting
	int numberOfFriends = 0;

	float2 dirVectForAFish = cuda_examples::normalize(velBeforeInteraction);

	float2 leftBorderOfVisibilityVector = float2();
	leftBorderOfVisibilityVector.x = dirVectForAFish.x * cosf(angleForFish) -
		dirVectForAFish.y * sinf(angleForFish);
	leftBorderOfVisibilityVector.y = dirVectForAFish.x * sinf(angleForFish) +
		dirVectForAFish.y * cosf(angleForFish);

	// All cosinuses that are less than this cosinus are not visible
	float cosBetweenBorderAndDirection = cuda_examples::dot(dirVectForAFish, leftBorderOfVisibilityVector);
	for (int i = 0; i < 4; i++)
	{
		if (cellsForSearch[i] != -1)
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

						float2 velOfFriend;
						velOfFriend.x = x_vel_before_movement[friendId];
						velOfFriend.y = y_vel_before_movement[friendId];

						float2 dirOfFriend = cuda_examples::normalize(velOfFriend);
						assert(cuda_examples::length(dirOfFriend) <= 1.0f + 10e-6 && cuda_examples::length(dirOfFriend) >= 1.0f - 10e-6);

						float2 vectToFriend = posOfFriend - fishPosition;
						assert(cuda_examples::length(vectToFriend) > 10e-6);

						float2 dirToFriend = cuda_examples::normalize(vectToFriend);

						// Check if friend is valid and fish can see it
						assert(cuda_examples::length(vectToFriend) >= 0);
						assert(cuda_examples::dot(dirVectForAFish, dirToFriend) >= -1.0f - 10e-6
							&& cuda_examples::dot(dirVectForAFish, dirToFriend) <= 1.0f + 10e-6);

						float distTofriend = cuda_examples::length(vectToFriend);
						if (distTofriend <= radiusForFish &&
							cuda_examples::dot(dirVectForAFish, dirToFriend) >= cosBetweenBorderAndDirection)
						{
							++numberOfFriends;
							// Alignment part
							if (distTofriend > separationRadius)
							{
								alignmentPart += velOfFriend;
								cohesionPart += posOfFriend;
								// Cohesion part
							}
							// Separation part
							if (distTofriend <= separationRadius)
							{
								separationPart -= vectToFriend;
							}
						}
					}
				}
			}
		}
	}

	float2 fearPower = float2();
	fearPower.x = 0.0f;
	fearPower.y = 0.0f;
	
	float2 cursorPos = float2();
	cursorPos.x = cursorX;
	cursorPos.y = cursorY;
	if (fearingWithCursor)
	{
		float2 vectToCursor = fishPosition - cursorPos;
		float len = cuda_examples::length(vectToCursor);
		assert(len >= 10e-6);
		if (len <= radiusForFish)
		{
			fearPower += vectToCursor;
			if (len <= separationRadius)
			{
				fearPower += vectToCursor;
			}
		}
	}

	if (numberOfFriends > 0) {
		alignmentPart = alignmentPart / numberOfFriends;
		cohesionPart = cohesionPart / numberOfFriends;
	}
	else {
		alignmentPart.x = 0.0f;
		alignmentPart.y = 0.0f;
		
		cohesionPart.x = 0.0f;
		cohesionPart.y = 0.0f;
	}
	
	float2 additionalVel = float2();
	additionalVel.x = 0.0f;
	additionalVel.y = 0.0f;

	additionalVel += alignmentNormal * alignmentPart;
	additionalVel += borderAvoidance;
	additionalVel += separationNormal * separationPart;
	additionalVel += cohesionNormal * (velBeforeInteraction - cohesionPart);
	additionalVel += fearKoef * fearPower;

	float2 velAfterCount = velBeforeInteraction + additionalVel;

	float valueOfVel = cuda_examples::length(velAfterCount);
	
	float2 directionVect = cuda_examples::normalize(velAfterCount);
	float eps = 10e-6;
	assert(abs(cuda_examples::length(velAfterCount) >= eps));
	assert(abs(cuda_examples::length(directionVect)) <= 1 + eps);
	if (valueOfVel > maxVel)
	{
		velAfterCount = directionVect * maxVel;
		assert(abs(cuda_examples::length(velAfterCount)) <= maxVel + eps);
	}
	else if (valueOfVel < minVel)
	{
		velAfterCount = directionVect * minVel;
		assert(abs(cuda_examples::length(velAfterCount)) <= minVel + eps);
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

__host__ __device__ void Fishes::FindTrianglesForAFish(int index, float* buffer)
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
	float2 first = 7 * direction + currentPosition;
	float2 second = 3 * normal + currentPosition;
	float2 third = 3 * -normal + currentPosition;
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
