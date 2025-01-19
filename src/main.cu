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

#define NUMBER_OF_FISHES 20
#define WIDTH 800
#define HEIGHT 600

#define THREAD_NUMBER 64

bool withGpu = true;

using namespace std;

const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec2 vertPos;\n"
"void main()\n"
"{\n"
" gl_Position = vec4(vertPos.x / 400, vertPos.y / 300, 0.0, 1.0);\n"
"}\0";

const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main()\n"
"{\n"
" FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);"
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
	h_fishes.GenerateTestFishes();
	for (int i = 0; i < NUMBER_OF_FISHES; i++)
	{
		printf("%f, ", h_fishes.x_before_movement[i]);
	}
	printf("\n");
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
	// I'll make it in other way. I'll create opengl buffer, with points for triangles.
	// In kernel or main function I'll pass the counted vertices to this buffer and after 
	// draw it
	GLuint VBO, VAO;
	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * NUMBER_OF_FISHES, nullptr, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	
	// Now i have registered buffer 
	cudaGraphicsResource* cuda_vbo_res;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_res, VBO, cudaGraphicsRegisterFlagsWriteDiscard));

	// TODO: Option struct
	Options h_options = Options();
	Options* d_options;
	checkCudaErrors(cudaMalloc((void**)&d_options, sizeof(Options)));


	Grid h_grid = Grid(NUMBER_OF_FISHES, h_options.radiusNormalFishes, WIDTH, HEIGHT, false);
	Grid d_grid = Grid(NUMBER_OF_FISHES, h_options.radiusNormalFishes, WIDTH, HEIGHT, true);
	h_grid.FindCellsForFishes(h_fishes);
	//d_grid.FindCellsForFishes(d_fishes);
	h_grid.SortCellsWithFishes();
	//d_grid.SortCellsWithFishes();
	h_grid.CleanStartsAndEnds();
	d_grid.CleanStartsAndEnds();
	h_grid.FindStartsAndEnds();
	//d_grid.FindStartsAndEnds();

	glUseProgram(shaderProgram);

	dim3 numBlocks((NUMBER_OF_FISHES + THREAD_NUMBER - 1) / THREAD_NUMBER);
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
			float* d_trianglesVertices = nullptr;
			size_t size = 0;
			checkCudaErrors(cudaGraphicsMapResources(1, &cuda_vbo_res));
			checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_trianglesVertices, &size, cuda_vbo_res));


			d_grid.FindCellsForFishes(d_fishes);
			d_grid.SortCellsWithFishes();
			d_grid.FindStartsAndEnds();
			// Count for every fish the next position and velocity
			CountForFishes << <numBlocks, THREAD_NUMBER >> > (d_grid, d_options, d_fishes, d_trianglesVertices, NUMBER_OF_FISHES);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
			d_grid.CleanStartsAndEnds();
			d_grid.CleanAfterAllCount(d_fishes);

			checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_vbo_res));
		}
		glUseProgram(shaderProgram);
		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, NUMBER_OF_FISHES * 3);


		// Render with opengl
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

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
