#pragma once 

struct __align__(16) Options
{
	int width = 1600;
	int height = 900;

	float separationNormalFishes = 1.0f;
	float alignmentNormalFishes = 1.0f;
	float cohesionNormalFishes = 1.0f;
	float minVelNormalFishes = 1.0f;
	float maxVelNormalFishes = 10.0f;
	float radiusNormalFishes = 40.0f;
	int angleNormalFishes = 60;
};