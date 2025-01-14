#pragma once 
#include "fishes.cuh"
#include <cuda_runtime.h>

struct Grid
{
private:
	bool onGpu;
	// Number of cells
	int n_x_cells;
	int n_y_cells;
	int n_cells;
	int n_fishes;
	int cellSize;
	int width;
	int height;
	void h_AllocateMemory();
	void d_AllocateMemory();

	void h_CleanMemory();
	void d_CleanMemory();
public:
	Grid(int nFishes, int radiusForFishes, int width, int heith, bool onGpu);
	~Grid();

	int* indices;
	int* cell_id;
	int* fish_id;

	// Starts and ends for cells
	int* cells_starts;
	int* cells_ends;

	// TODO: Init Variables
	void FindCellsForFishes(Fishes fishes);
	void SortCellsWithFishes();
	void FindStartsAndEnds();
};

struct CellForFishFunctor
{
private:
	float* xPosition;
	float* yPosition;
	float sizeOfCell;
	float width;
	float height;
public:
	CellForFishFunctor(float* x_pos, float* y_pos, float sizeOfCell, float widht, float height) :
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

struct FindStartsAndEndsFunctor
{
private:
	int n_fishes;
	int* startsOfCells;
	int* endsOfCells;
	int* sortedCells;
public: FindStartsAndEndsFunctor(int nFishes, int* startsOfCells, int* endsOfCells, int* sortedCells):
	n_fishes(nFishes), startsOfCells(startsOfCells),
		endsOfCells(endsOfCells), sortedCells(sortedCells) {}

	__host__ __device__ int operator()(int& index)
	{

		if (index == 0 || sortedCells[index - 1] != sortedCells[index])
		{
			startsOfCells[sortedCells[index]] = index;
		}
		else if (index == n_fishes - 1 || sortedCells[index + 1] != sortedCells[index])
		{
			endsOfCells[sortedCells[index]] = index;
		}
		return index;
	}
};

