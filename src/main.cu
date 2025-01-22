#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "../objects/fishes.cuh"
#include "../objects/grid.cuh"
#include "../objects/options.cuh"
#include "../include/helpers.cuh"
#include <cuda_gl_interop.h>
#include "../main_loop/main_loop_gpu.cuh"
#include <vector>

#define NUMBER_OF_FISHES 20
#define WIDTH 1600
#define HEIGHT 900

#define NUMBER_OF_POINTS_FOR_TRIANGLE 32

#define THREAD_NUMBER 128

bool withGpu = true;

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

int main()
{
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



	Fishes h_fishes = Fishes(NUMBER_OF_FISHES, false);
	//h_fishes.GenerateTestFishes();
	h_fishes.GenerateRandomFishes(WIDTH, HEIGHT, 1.0f, 4.0f);

	Fishes d_fishes = Fishes(NUMBER_OF_FISHES, true);
	d_fishes.d_CopyFishesFromCPU(h_fishes.x_before_movement, h_fishes.y_before_movement,
		h_fishes.x_vel_before_movement, h_fishes.y_vel_before_movement, h_fishes.types);


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
	GLuint VBO_Centers, VAO_Triangles;
	glGenBuffers(1, &VBO_Centers);
	glGenVertexArrays(1, &VAO_Triangles);
	glBindVertexArray(VAO_Triangles);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_Centers);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * NUMBER_OF_FISHES, nullptr, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	
	cudaGraphicsResource* cuda_vbo_res_triangles;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_res_triangles, VBO_Centers, cudaGraphicsRegisterFlagsWriteDiscard));

	GLuint VBO_Circles, VAO_Circles;
	glGenBuffers(1, &VBO_Circles);
	glGenVertexArrays(1, &VAO_Circles);
	glBindVertexArray(VAO_Circles);
	glBindBuffer(GL_ARRAY_BUFFER, VBO_Circles);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * NUMBER_OF_POINTS_FOR_TRIANGLE * 2 * NUMBER_OF_FISHES, nullptr, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);

	cudaGraphicsResource* cuda_vbo_res_circle;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_res_circle, VBO_Circles, cudaGraphicsRegisterFlagsWriteDiscard));
	
	GLint firsts[NUMBER_OF_FISHES];
	GLsizei count[NUMBER_OF_FISHES];

	for (int i = 0; i < NUMBER_OF_FISHES; i++)
	{
		firsts[i] = i * NUMBER_OF_POINTS_FOR_TRIANGLE;
		count[i] = NUMBER_OF_POINTS_FOR_TRIANGLE;
	}

	// TODO: Option struct
	Options h_options = Options();
	Options* d_options;
	checkCudaErrors(cudaMalloc((void**)&d_options, sizeof(Options)));
	checkCudaErrors(cudaMemcpy(d_options, &h_options, sizeof(Options), cudaMemcpyHostToDevice));

	Grid h_grid = Grid(NUMBER_OF_FISHES, h_options.radiusNormalFishes, WIDTH, HEIGHT, false);
	Grid d_grid = Grid(NUMBER_OF_FISHES, h_options.radiusNormalFishes, WIDTH, HEIGHT, true);
	h_grid.h_InitialiseArraysIndicesAndFishes();
	d_grid.d_InitialiseArraysIndicesAndFishes(h_grid.indices);

	h_grid.FindCellsForFishes(h_fishes);
	h_grid.SortCellsWithFishes();
	h_grid.CleanStartsAndEnds();
	h_grid.FindStartsAndEnds();


	dim3 numBlocks((NUMBER_OF_FISHES + THREAD_NUMBER - 1) / THREAD_NUMBER);

	cudaEvent_t start_main, stop_main;
	cudaEventCreate(&start_main);
	cudaEventCreate(&stop_main);

	cudaEvent_t start_circles, stop_circles;
	cudaEventCreate(&start_circles);
	cudaEventCreate(&stop_circles);
	d_grid.CleanStartsAndEnds();
	while (!glfwWindowShouldClose(window))
	{
		// Imgui window
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::ShowDemoWindow();


		glClear(GL_COLOR_BUFFER_BIT);

		float* d_trianglesVertices = nullptr;
		float* d_circlesVertices = nullptr;
		if (withGpu)
		{
			size_t size = 0;
			checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_res_triangles));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_trianglesVertices, &size, cuda_vbo_res_triangles));
			checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_res_circle));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_circlesVertices, &size, cuda_vbo_res_circle));
			cudaEventRecord(start_main);

			d_grid.FindCellsForFishes(d_fishes);
			d_grid.SortCellsWithFishes();
			d_grid.FindStartsAndEnds();
			// Count for every fish the next position and velocity
			CountForFishes << <numBlocks, THREAD_NUMBER >> > (d_grid, d_options, d_fishes, d_trianglesVertices, NUMBER_OF_FISHES);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
			d_grid.CleanStartsAndEnds();
			d_grid.CleanAfterAllCount(d_fishes);
			cudaEventRecord(stop_main);
			float milliseconds = 0;
			cudaEventSynchronize(stop_main);
			cudaEventElapsedTime(&milliseconds, start_main, stop_main);
			printf("%f milliseconds for a frame\n", milliseconds);

			cudaEventRecord(start_circles);
			CountCircleForFish << <numBlocks, THREAD_NUMBER >> > (d_fishes, d_circlesVertices, NUMBER_OF_FISHES, NUMBER_OF_POINTS_FOR_TRIANGLE,
				h_options.radiusNormalFishes);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
			cudaEventRecord(stop_circles);
			float milliseconds_circle;
			cudaEventSynchronize(stop_circles);
			cudaEventElapsedTime(&milliseconds_circle, start_circles, stop_circles);
			printf("%f milliseconds for the circles\n", milliseconds_circle);

			checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_res_triangles));
			checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_res_circle));
		}

		
		int screenWidth = glGetUniformLocation(shaderProgram, "width");
		int screenHeight = glGetUniformLocation(shaderProgram, "height");
		int colorPosition = glGetUniformLocation(shaderProgram, "color");
		glUseProgram(shaderProgram);
		glUniform1f(screenWidth, WIDTH);
		glUniform1f(screenHeight, HEIGHT);
		glUniform4f(colorPosition, 0.0f, 1.0f, 0.0f, 1.0f);

		glBindVertexArray(VAO_Triangles);
		glDrawArrays(GL_TRIANGLES, 0, NUMBER_OF_FISHES * 3);

		glUniform4f(colorPosition, 1.0f, 1.0f, 1.0f, 1.0f);
		glBindVertexArray(VAO_Circles);
		glMultiDrawArrays(GL_LINE_LOOP, firsts, count, NUMBER_OF_FISHES);

		// Render with opengl
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
		glfwPollEvents();
		//free(test_array);
	}
	cudaEventDestroy(start_main);
	cudaEventDestroy(stop_main);
	cudaEventDestroy(start_circles);

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
