#pragma once

struct Fishes
{
	enum FishType
	{
		NormalFish,
		LeaderOfNormalFishes,
		Predator
	};
	int N;

	// Radius of visibility
	float radius;
	// Angle of visibility
	float angle;

	float* x_before_movement;
	float* y_before_movement;
	float* x_vel_before_movement;
	float* y_vel_before_movement;
	 
	float* x_after_movement;
	float* y_after_movement;
	float* x_vel_after_movement;
	float* y_vel_after_movement;

	FishType* types;

	void h_allocate_memory_for_fishes();
	void d_allocate_memory_for_fishes();
	void h_clean_memory_for_fishes();
	void d_clean_memory_for_fishes();

	enum FishType
	{
		NormalFish,
		LeaderOfNormalFishes,
		Predator
	};
};