cmake_minimum_required(VERSION 3.12)

project(1_drawing_a_triangle)

set(CMAKE_CXX_STANDARD 17)

add_executable(main_drawing_a_triangle main.cpp)
target_link_libraries(main_drawing_a_triangle PRIVATE
    glm::glm
    glfw
    ${Vulkan_LIBRARIES}
)
