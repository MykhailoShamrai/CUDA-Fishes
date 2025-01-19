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

#define NUMBER_OF_FISHES 20
#define WIDTH 800
#define HEIGHT 600

#define THREADS_PER_BLOCK 64

bool withGpu = true;

using namespace std;


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
	h_fishes.GenerateTestFishes();
	for (int i = 0; i < NUMBER_OF_FISHES; i++)
	{
		printf("%f, ", h_fishes.x_before_movement[i]);
	}
	printf("\n");
	Fishes d_fishes = Fishes(NUMBER_OF_FISHES, true);
	d_fishes.d_CopyFishesFromCPU(h_fishes.x_before_movement, h_fishes.y_before_movement,
		h_fishes.x_vel_before_movement, h_fishes.y_vel_before_movement, h_fishes.types);

	// OpenGl stuff
	// I'll make it in other way. I'll create opengl buffer, with points for triangles.
	// In kernel or main function I'll pass the counted vertices to this buffer and after 
	// draw it
	GLuint VBO, VAO;
	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * NUMBER_OF_FISHES, nullptr, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	
	// Now i have registered buffer 
	cudaGraphicsResource* cuda_vbo_res;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_res, VBO, cudaGraphicsRegisterFlagsWriteDiscard));

	float* d_trianglesVertices;
	size_t size;
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_res));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)d_trianglesVertices, &size, cuda_vbo_res));


	// TODO: Option struct
	Options h_options = Options();
	Options* d_options;
	checkCudaErrors(cudaMalloc((void**)&d_options, sizeof(Options)));


	Grid h_grid = Grid(NUMBER_OF_FISHES, h_options.radiusNormalFishes, WIDTH, HEIGHT, false);
	Grid d_grid = Grid(NUMBER_OF_FISHES, h_options.radiusNormalFishes, WIDTH, HEIGHT, true);
	h_grid.FindCellsForFishes(h_fishes);
	d_grid.FindCellsForFishes(d_fishes);
	h_grid.SortCellsWithFishes();
	d_grid.SortCellsWithFishes();
	h_grid.CleanStartsAndEnds();
	d_grid.CleanStartsAndEnds();
	h_grid.FindStartsAndEnds();
	d_grid.FindStartsAndEnds();

	while (!glfwWindowShouldClose(window))
	{
		// Imgui window
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::ShowDemoWindow();

		glClear(GL_COLOR_BUFFER_BIT);

		if (withGpu)
		{
			dim3 numBlocks((NUMBER_OF_FISHES + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
		}

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	d_fishes.d_CleanMemoryForFishes();
	h_fishes.h_CleanMemoryForFishes();
	d_grid.d_CleanMemory();
	h_grid.h_CleanMemory();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwTerminate();
	return 0;
}
