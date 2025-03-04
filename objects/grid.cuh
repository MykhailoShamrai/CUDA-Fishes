#pragma once 
#include <cuda_runtime.h>
#include <thrust/functional.h>

struct Fishes;

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

public:
	Grid(int nFishes, int radiusForFishes, int width, int heith, bool onGpu);
	~Grid() {};

	int* indices;
	int* cell_id;
	int* quarter_number;
	int* fish_id;
	// Starts and ends for cells
	int* cells_starts;
	int* cells_ends;

	void h_CleanMemory();
	void d_CleanMemory();
	void h_InitialiseArraysIndicesAndFishes();
	void d_InitialiseArraysIndicesAndFishes(int* initialisedIndexArray);
	void FindCellsForFishes(Fishes& fishes);
	void SortCellsWithFishes();
	void FindStartsAndEnds();
	void CleanStartsAndEnds();
	void CleanAfterAllCount(Fishes& fishes);
	__host__ __device__ int ReturnNumberOfCells() { return n_cells; };
	__host__ __device__ int ReturnNumberOfCellsX() { return n_x_cells; };
	__host__ __device__ int ReturnNumberOfCellsY() { return n_y_cells; };
	__host__ __device__ int ReturnWidth() { return width; };
	__host__ __device__ int ReturnHeight() { return height; };
};

struct CellForFishFunctor
{
private:
	float* xPosition;
	float* yPosition;
	float sizeOfCell;
	float width;
	float height;
	int nXCells;
	int nYCells;
public:
	CellForFishFunctor(float* x_pos, float* y_pos, float sizeOfCell, float width, float height, int nXCells, int nYCells) :
		xPosition(x_pos), yPosition(y_pos), sizeOfCell(sizeOfCell), width(width), height(height), nXCells(nXCells), 
	nYCells(nYCells){};

	__host__ __device__ int operator()(int index)
	{
		float x = xPosition[index];
		float y = yPosition[index];
		float transformed_x = x + width / 2;
		float transformed_y = -y + height / 2;
		int x_index = (transformed_x) / sizeOfCell;
		int y_index = (transformed_y) / sizeOfCell;
		return x_index + nXCells * y_index;
	}
};

struct QuarterForFishFunctor
{
private:
	float* xPosition;
	float* yPosition;
	int* cellId;
	float sizeOfCell;
	int width;
	int height;
	int nXCells;
	int nYCells;
public:
	QuarterForFishFunctor(float* x_pos, float* y_pos, int* cellId, int sizeofCell, int width,
		int height, int nXCells, int nYCells): xPosition(x_pos), yPosition(y_pos), cellId(cellId),
		sizeOfCell(sizeofCell), width(width), height(height), nXCells(nXCells), nYCells(nYCells){}

	__host__ __device__ int operator()(int index)
	{
		float x = xPosition[index];
		float y = yPosition[index];
		float transformed_x = x + width / 2;
		float transformed_y = -y + height / 2;
		int idOfCell = cellId[index];
		int i = idOfCell % nXCells;
		int j = idOfCell / nXCells;
		int xCellStart = i * sizeOfCell;
		int yCellStart = j * sizeOfCell;
		// Quarters 2 and 3
		if (transformed_x <= (float)xCellStart + (float)sizeOfCell / 2)
		{
			if (transformed_y <= (float)yCellStart + (float)sizeOfCell / 2)
			{
				return 2;
			}
			else
			{
				return 3;
			}
		}
		// Quarters 1 and 4
		else
		{
			if (transformed_y <= (float)yCellStart + (float)sizeOfCell / 2)
			{
				return 1;
			}
			else
			{
				return 4;
			}
		}
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

	__host__ __device__ int operator()(int index)
	{
		if (index == 0 || sortedCells[index - 1] != sortedCells[index])
		{
			startsOfCells[sortedCells[index]] = index;
		}
		if (index == n_fishes - 1 || sortedCells[index + 1] != sortedCells[index])
		{
			endsOfCells[sortedCells[index]] = index;
		}
		return index;
	}
};

struct InitArraysFunctor
{	
	__host__ __device__ int operator()(int index)
	{
		return 1;
	}
};

struct CleanStartsAndEndsFunctor
{
	__host__ __device__ int operator()(int index)
	{
		return -1;
	}
};
