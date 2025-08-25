cmake_minimum_required(VERSION 3.12)

project(vulkan_tutorial)

set(CMAKE_CXX_STANDARD 17)

# vulkan
find_package(Vulkan REQUIRED)
include_directories(${Vulkan_INCLUDE_DIRS})

# glslc
find_program(
    GLSL_COMPILER glslc
    HITNS "$ENV{VULKAN_SDK}/bin"
)
if(NOT GLSL_COMPILER)
    message(FATAL_ERROR "Cannot find glslc")
else()
    message("Found glslc")
endif()

# glfw
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("third_party/glfw-3.4/")

# stb_image
include_directories("third_party/stb")

# glm
add_subdirectory("third_party/glm-1.0.1/")

# 0. environment setup
add_subdirectory("src/0_setup_environment/")

# 1. drawing a triangle
add_subdirectory("src/1_drawing_a_triangle")

# 2. vertex buffers
add_subdirectory("src/2_vertex_buffers")

# 3. uniform buffers
add_subdirectory("src/3_uniform_buffers")

# 4. texture mappings
add_subdirectory("src/4_texture_mappings")

# 5. depth buffering
add_subdirectory("src/5_depth_buffering")