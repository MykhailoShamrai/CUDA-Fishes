# CMakeList.txt : CMake project for CUDA-Fishes, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (CUDA-Fishes "main.cu")

target_include_directories(CUDA-Fishes PRIVATE 
	../third_party/glfw)
target_link_libraries(CUDA-Fishes PRIVATE 
	glfw glad ${CMAKE_CUDA_LIBRARIES})
target_include_directories(CUDA-Fishes PRIVATE
	"../third_party/glad/include" "../third_party/glfw/include")

set_target_properties(${PROJECT_NAME} PROPERTIES
CUDA_SEPARABLE_COMPILATION ON
CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
)

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CUDA-Fishes PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
