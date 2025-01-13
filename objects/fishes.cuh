#pragma once

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
};