﻿# CMakeList.txt : CMake project for CUDA-Fishes, include source and define
# project specific logic here.
#

# Add source to this project's executable.
set(IMGUI_SOURCE_FILES	../third_party/imgui/imgui_impl_glfw.cpp
						../third_party/imgui/imgui_impl_glfw.h
						../third_party/imgui/imgui_impl_opengl3.cpp
						../third_party/imgui/imgui_impl_opengl3.h
						../third_party/imgui/imgui_impl_opengl3_loader.h
						../third_party/imgui/imgui_draw.cpp
						../third_party/imgui/imgui_internal.h
						../third_party/imgui/imgui_stdlib.cpp
						../third_party/imgui/imgui_stdlib.h
						../third_party/imgui/imgui.cpp
						../third_party/imgui/imgui.h
						../third_party/imgui/imgui_demo.cpp
						../third_party/imgui/imgui_tables.cpp
						../third_party/imgui/imgui_widgets.cpp
						../third_party/imgui/imconfig.h
						../third_party/imgui/imstb_rectpack.h
						../third_party/imgui/imstb_textedit.h
						../third_party/imgui/imstb_truetype.h)

add_executable (CUDA-Fishes "main.cu"
					${IMGUI_SOURCE_FILES}
					"../includes/helpers.cu"
					"../objects/fishes.cu"
					"../objects/grid.cu"
					"../main_loop/main_loop_gpu.cu"
					"../main_loop/main_loop_cpu.cu")

target_include_directories(CUDA-Fishes PRIVATE 
	"../third_party/glfw/include"
	"../third_party/glad/include"
	../third_party/imgui
	../third_party/cuda-samples
	../includes
	../objects)


find_package("OpenGL" REQUIRED)

target_link_libraries(CUDA-Fishes PRIVATE 
	glfw glad ${OPENGL_gl_LIBRARY} ${CMAKE_CUDA_LIBRARIES})


set_target_properties(${PROJECT_NAME} PROPERTIES
CUDA_SEPARABLE_COMPILATION ON

)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CUDA-Fishes PROPERTY CXX_STANDARD 20)
endif()

