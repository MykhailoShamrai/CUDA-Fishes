# CMakeList.txt : CMake project for CUDA-Fishes, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (CUDA-Fishes "CUDA-Fishes.cpp" "CUDA-Fishes.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET CUDA-Fishes PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
