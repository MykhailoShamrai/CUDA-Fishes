﻿#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "../objects/fishes.cuh"
#include "../objects/grid.cuh"
#include "../objects/options.cuh"
#include "../includes/helpers.cuh"
#include <cuda_gl_interop.h>
#include "../main_loop/main_loop_gpu.cuh"
#include "../main_loop/main_loop_cpu.cuh"
#include <vector>
#include <string>

#define WIDTH 1600
#define HEIGHT 900
static int screen_width = WIDTH;
static int screen_height = HEIGHT;

#define NUMBER_OF_POINTS_FOR_CIRCLE 32

#define THREAD_NUMBER 128

static int numberOfFishes = 100;
float cursorPosX = 0.0f;
float cursorPosY = 0.0f;

static bool fearingWithCursor = false;
static bool withGpu = true;
static bool withGpuChanged = false;
static bool circleDrawing = false;
static bool isSimulationRunning = true;

using namespace std;

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec2 vertPos;\n"
"uniform float width;\n"
"uniform float height;\n"
"void main()\n"
"{\n"
" gl_Position = vec4(vertPos.x / width * 2, vertPos.y / height * 2, 0.0, 1.0);\n"
"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"uniform vec4 color;\n"
"void main()\n"
"{\n"
" FragColor = color;\n"
"}\0";

static void key_press_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_C && action == GLFW_PRESS)
	{
		circleDrawing = !circleDrawing;
	}
	else if (key == GLFW_KEY_G && action == GLFW_PRESS && isSimulationRunning)
	{
		withGpuChanged = true;
	}
	else if (key == GLFW_KEY_F && action == GLFW_PRESS)
	{
		fearingWithCursor = !fearingWithCursor;
	}
	else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
	{
		isSimulationRunning = !isSimulationRunning;
	}
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
	cursorPosX = (float)xpos - WIDTH / 2;
	cursorPosY = -(float)ypos + HEIGHT / 2;
}

static void framebuffer_size_callback(GLFWwindow*, int new_width, int new_height)
{
	screen_width = new_width;
	screen_height = new_height;
	glViewport(0, 0, new_width, new_height);
}

char* get_cmd_option(char** begin, char** end, const std::string& option)
{
	char** itr = std::find(begin, end, option);
	if (itr != end && ++itr != end)
	{
		return *itr;
	}
	return 0;
}

bool cmd_option_exists(char** begin, char** end, const std::string& option)
{
	return std::find(begin, end, option) != end;
}

int main(int argc, char** argv)
{
	if (cmd_option_exists(argv, argv + argc, "-n"))
	{
		try
		{
			char* val = get_cmd_option(argv, argv + argc, "-n");
			int num_fishes_tmp = std::stoi(val, 0);
			if (num_fishes_tmp <= 0)
				throw std::invalid_argument("Received negative value or 0!");
			numberOfFishes = num_fishes_tmp;
		}
		catch (const std::exception& ex)
		{
			fprintf(stderr, "Wrong argument for number of fishes! %s Default value = %d is set\n", ex.what(), numberOfFishes);
		}
	}

	// Initialization
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	#ifdef __APPLE__
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);	
	#endif

	GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Fishes", NULL, NULL);
	if (window == NULL)
	{
		cout << "Failed to create GLFW window" << endl;
		glfwTerminate();
		return -1;
	}

	glfwMakeContextCurrent(window);
	glfwSetKeyCallback(window, key_press_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		cout << "Failed to initialize GLAD" << endl;
		return -1;
	}

	// ImGui initialization
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init();

	Fishes h_fishes = Fishes(numberOfFishes, false);
	h_fishes.GenerateRandomFishes(WIDTH, HEIGHT, 1.0f, 4.0f);

	Fishes d_fishes = Fishes(numberOfFishes, true);
	d_fishes.d_CopyFishesFromCPU(h_fishes);


	int success;
	char infoLog[512];

	unsigned int vertexShader;
	vertexShader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexShader, 1, &vertexShaderSource, nullptr);
	glCompileShader(vertexShader);

	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		glGetShaderInfoLog(vertexShader, 512, nullptr, infoLog);
		std::cout << "ERRORR::SHADER::VERTEX::COMPILATION_FAILED\n" <<
			infoLog << std::endl;
	}
	
	unsigned int fragmentShader;
	fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);

	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);

	if (!success)
	{
		glGetShaderInfoLog(fragmentShader, 512, nullptr, infoLog);
		std::cout << "ERRORR::SHADER::FRAGMENT::COMPILATION_FAILED\n" <<
			infoLog << std::endl;
	}

	unsigned int shaderProgram;
	shaderProgram = glCreateProgram();
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);

	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
		std::cout << "ERRORR::SHADER::PROGRAM::LINKING_FAILED\n" <<
			infoLog << std::endl;
	}
	// OpenGl stuff
	// Main triangle that represents a fish
	GLuint VBO_Triangles_GPU, VAO_Triangles_GPU;
	glGenBuffers(1, &VBO_Triangles_GPU);
	glGenVertexArrays(1, &VAO_Triangles_GPU);
	glBindVertexArray(VAO_Triangles_GPU);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_Triangles_GPU);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * numberOfFishes, nullptr, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	
	cudaGraphicsResource* cuda_vbo_res_triangles;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_res_triangles, VBO_Triangles_GPU, cudaGraphicsRegisterFlagsWriteDiscard));

	GLuint VBO_Circles_GPU, VAO_Circles_GPU;
	glGenBuffers(1, &VBO_Circles_GPU);
	glGenVertexArrays(1, &VAO_Circles_GPU);
	glBindVertexArray(VAO_Circles_GPU);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_Circles_GPU);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * NUMBER_OF_POINTS_FOR_CIRCLE * 2 * numberOfFishes, nullptr, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	cudaGraphicsResource* cuda_vbo_res_circle;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_res_circle, VBO_Circles_GPU, cudaGraphicsRegisterFlagsWriteDiscard));
	
	GLint* firsts = (GLint*)malloc(sizeof(GLint) * numberOfFishes);
	GLsizei* count = (GLsizei*)malloc(sizeof(GLsizei) * numberOfFishes);

	for (int i = 0; i < numberOfFishes; i++)
	{
		firsts[i] = i * NUMBER_OF_POINTS_FOR_CIRCLE;
		count[i] = NUMBER_OF_POINTS_FOR_CIRCLE;
	}

	float* h_triangles_buffer = (float*)malloc(sizeof(float) * 6 * numberOfFishes);
	float* h_circles_buffer = (float*)malloc(sizeof(float) * 2 * NUMBER_OF_POINTS_FOR_CIRCLE * numberOfFishes);

	GLuint VBO_Triangles_CPU, VAO_Triangles_CPU;
	glGenBuffers(1, &VBO_Triangles_CPU);
	glGenVertexArrays(1, &VAO_Triangles_CPU);
	glBindVertexArray(VAO_Triangles_CPU);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_Triangles_CPU);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * numberOfFishes, nullptr, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	GLuint VBO_Circles_CPU, VAO_Circles_CPU;
	glGenBuffers(1, &VBO_Circles_CPU);
	glGenVertexArrays(1, &VAO_Circles_CPU);
	glBindVertexArray(VAO_Circles_CPU);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_Circles_CPU);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * NUMBER_OF_POINTS_FOR_CIRCLE * 2 * numberOfFishes, h_circles_buffer, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	Options h_options = Options();
	Options* d_options;
	checkCudaErrors(cudaMalloc((void**)&d_options, sizeof(Options)));
	checkCudaErrors(cudaMemcpy(d_options, &h_options, sizeof(Options), cudaMemcpyHostToDevice));

	Grid h_grid = Grid(numberOfFishes, h_options.radiusForFishes, WIDTH, HEIGHT, false);
	Grid d_grid = Grid(numberOfFishes, h_options.radiusForFishes, WIDTH, HEIGHT, true);
	h_grid.h_InitialiseArraysIndicesAndFishes();
	d_grid.d_InitialiseArraysIndicesAndFishes(h_grid.indices);



	dim3 numBlocks((numberOfFishes + THREAD_NUMBER - 1) / THREAD_NUMBER);

	cudaEvent_t start_main, stop_main;
	cudaEventCreate(&start_main);
	cudaEventCreate(&stop_main);

	cudaEvent_t start_circles, stop_circles;
	cudaEventCreate(&start_circles);
	cudaEventCreate(&stop_circles);
	h_grid.CleanStartsAndEnds();
	d_grid.CleanStartsAndEnds();
	bool valuesChanged = false;
	while (!glfwWindowShouldClose(window))
	{
		// Imgui window
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		// Here I also can print metrics
		ImGui::Text("Options for fishes:");
		valuesChanged |= ImGui::SliderFloat("Separation", &h_options.separationForFishes, 0.01f, 1.0f);
		valuesChanged |= ImGui::SliderFloat("Alignment", &h_options.alignmentForFishes, 0.01f, 1.0f);
		valuesChanged |= ImGui::SliderFloat("Cohesion", &h_options.cohesionForFishes, 0.00001f, 0.001f, "%.5f");

		valuesChanged |= ImGui::SliderFloat("Max Velocity", &h_options.maxVelFishes, 0.7f, 5.0f);
		valuesChanged |= ImGui::SliderFloat("Inner radius", &h_options.radiusSeparation, 5.0f, h_options.radiusForFishes);

		valuesChanged |= ImGui::SliderFloat("Force for wall avoidance", &h_options.forceForWallAvoidance, 0.1f, 0.5f);
		valuesChanged |= ImGui::SliderFloat("Range to border", &h_options.rangeToBorderToStartTurn, 50.0f, 200.0f);
		valuesChanged |= ImGui::SliderFloat("Fear of cursor", &h_options.powerOfFearForCursor, 0.01f, 0.1f, "%.3f");

		ImGui::Checkbox("Is animation running", &isSimulationRunning);
		ImGui::BeginDisabled();
		ImGui::Checkbox("GPU", &withGpu);
		ImGui::EndDisabled();
		if (!isSimulationRunning)
		{
			ImGui::BeginDisabled();
		}
		if (ImGui::Button("Switch"))
		{
			withGpuChanged = true;
		}
		if (!isSimulationRunning)
		{
			ImGui::EndDisabled();
		}

		ImGui::Checkbox("Drawing of circles", &circleDrawing);
		ImGui::Checkbox("Cursor interaction", &fearingWithCursor);
		if (ImGui::Button("Reset to Defaults"))
		{
			h_options.resetToDefaults();
			valuesChanged = true;
		}
		float fps = ImGui::GetIO().Framerate;
		ImGui::Text("FPS: %.1f", fps);


		if (valuesChanged)
		{
			valuesChanged = false;
			checkCudaErrors(cudaMemcpy(d_options, &h_options, sizeof(Options), cudaMemcpyHostToDevice));
		}



		glClear(GL_COLOR_BUFFER_BIT);

		float* d_trianglesVertices = nullptr;
		float* d_circlesVertices = nullptr;

		// iF state of withGpu was changed it's necessary to copy from gpu to cpu last data or in other side
		if (withGpuChanged)
		{
			withGpuChanged = false;
			withGpu = !withGpu;
			if (withGpu)
			{
				d_fishes.d_CopyFishesFromCPU(h_fishes);
			}
			else
			{
				h_fishes.h_CopyFishesFromGPU(d_fishes);
			}
		}


		if (withGpu)
		{
			size_t size = 0;
			if (isSimulationRunning)
			{
				checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_res_triangles));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_trianglesVertices, &size, cuda_vbo_res_triangles));
				cudaEventRecord(start_main);

				d_grid.FindCellsForFishes(d_fishes);
				d_grid.SortCellsWithFishes();
				d_grid.FindStartsAndEnds();
				// Count for every fish the next position and velocity
				CountForFishesGpu << <numBlocks, THREAD_NUMBER >> > (d_grid, d_options, d_fishes, d_trianglesVertices, numberOfFishes,
					cursorPosX, cursorPosY, fearingWithCursor);
				checkCudaErrors(cudaGetLastError());
				checkCudaErrors(cudaDeviceSynchronize());
				d_grid.CleanStartsAndEnds();
				d_grid.CleanAfterAllCount(d_fishes);
				cudaEventRecord(stop_main);
				float milliseconds = 0.0f;
				cudaEventSynchronize(stop_main);
				cudaEventElapsedTime(&milliseconds, start_main, stop_main);
				ImGui::Text("milliseconds for main kernel: %.4f", milliseconds);
				checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_res_triangles));
			}

			// Main drawing
			int screenWidth = glGetUniformLocation(shaderProgram, "width");
			int screenHeight = glGetUniformLocation(shaderProgram, "height");
			int colorPosition = glGetUniformLocation(shaderProgram, "color");
			glUseProgram(shaderProgram);
			glUniform1f(screenWidth, WIDTH);
			glUniform1f(screenHeight, HEIGHT);
			glUniform4f(colorPosition, 0.0f, 1.0f, 0.0f, 1.0f);


			glBindVertexArray(VAO_Triangles_GPU);
			glDrawArrays(GL_TRIANGLES, 0, numberOfFishes * 3);

			// Optional drawing of circles around
			if (circleDrawing)
			{
				checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_res_circle));
				checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_circlesVertices, &size, cuda_vbo_res_circle));

				cudaEventRecord(start_circles);
				CountCircleForFishesGpu << <numBlocks, THREAD_NUMBER >> > (d_fishes, d_circlesVertices,
					numberOfFishes, NUMBER_OF_POINTS_FOR_CIRCLE, h_options.radiusForFishes);

				checkCudaErrors(cudaGetLastError());
				checkCudaErrors(cudaDeviceSynchronize());
				cudaEventRecord(stop_circles);
				float milliseconds_circle = 0.0f;
				cudaEventSynchronize(stop_circles);
				cudaEventElapsedTime(&milliseconds_circle, start_circles, stop_circles);
				ImGui::Text("milliseconds for circles generation: %.4f", milliseconds_circle);
				checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_res_circle));

				glUniform4f(colorPosition, 1.0f, 1.0f, 1.0f, 1.0f);
				glBindVertexArray(VAO_Circles_GPU);
				glMultiDrawArrays(GL_LINE_LOOP, firsts, count, numberOfFishes);
			}

		}
		else
		{
			if (isSimulationRunning)
			{
				double startOfFunction = glfwGetTime();
				h_grid.FindCellsForFishes(h_fishes);
				h_grid.SortCellsWithFishes();
				h_grid.FindStartsAndEnds();

				CountForFishesCpu(h_grid, h_options, h_fishes, h_triangles_buffer, numberOfFishes, cursorPosX, cursorPosY,
					fearingWithCursor);

				h_grid.CleanStartsAndEnds();
				h_grid.CleanAfterAllCount(h_fishes);

				double endOfFunction = glfwGetTime();
				double elapsedTime = (endOfFunction - startOfFunction) * 1000;
				ImGui::Text("milliseconds for main function: %.4f", elapsedTime);
			}

			int screenWidth = glGetUniformLocation(shaderProgram, "width");
			int screenHeight = glGetUniformLocation(shaderProgram, "height");
			int colorPosition = glGetUniformLocation(shaderProgram, "color");
			glUseProgram(shaderProgram);
			glUniform1f(screenWidth, WIDTH);
			glUniform1f(screenHeight, HEIGHT);
			glUniform4f(colorPosition, 0.0f, 1.0f, 0.0f, 1.0f);


			glBindBuffer(GL_ARRAY_BUFFER, VBO_Triangles_CPU);
			glBindVertexArray(VAO_Triangles_CPU);
			glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * 6 * numberOfFishes, h_triangles_buffer);
			glDrawArrays(GL_TRIANGLES, 0, numberOfFishes * 3);


			if (circleDrawing)
			{
				double startOfCirclesGeneration = glfwGetTime();
				CountCircleForFishesCpu(h_fishes, h_circles_buffer, numberOfFishes, NUMBER_OF_POINTS_FOR_CIRCLE, h_options.radiusForFishes);
				double endOfCirclesGeneration = glfwGetTime();
				double elapsedTime = (endOfCirclesGeneration - startOfCirclesGeneration) * 1000;
				ImGui::Text("milliseconds for circles generation: %.4f", elapsedTime);
				glUniform4f(colorPosition, 1.0f, 1.0f, 1.0f, 1.0f);
				glBindBuffer(GL_ARRAY_BUFFER, VBO_Circles_CPU);
				glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * NUMBER_OF_POINTS_FOR_CIRCLE * 2 * numberOfFishes, h_circles_buffer);
				glBindVertexArray(VAO_Circles_CPU);
				glMultiDrawArrays(GL_LINE_LOOP, firsts, count, numberOfFishes);
			}
		}
		

	
		// Render with opengl
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
		glfwPollEvents();
	}
	cudaEventDestroy(start_main);
	cudaEventDestroy(stop_main);
	cudaEventDestroy(start_circles);

	free(firsts);
	free(count);

	d_fishes.d_CleanMemoryForFishes();
	h_fishes.h_CleanMemoryForFishes();
	d_grid.d_CleanMemory();

	h_grid.h_CleanMemory();
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();
	return 0;
}
