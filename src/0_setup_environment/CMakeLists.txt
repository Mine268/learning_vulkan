cmake_minimum_required(VERSION 3.12)

project(0_setup_environment)

set(CMAKE_CXX_STANDARD 17)

add_executable(main_setup_environment main.cpp)
target_link_libraries(main_setup_environment PRIVATE
    glm::glm
    glfw
    ${Vulkan_LIBRARIES}
)