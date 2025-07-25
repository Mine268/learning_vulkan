cmake_minimum_required(VERSION 3.12)

project(2_vertex_buffers)

set(CMAKE_CXX_STANDARD 17)

# shaders
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/shaders)

set(SHADER_SOURCES
    shaders/shader.frag
    shaders/shader.vert
)

foreach(shader ${SHADER_SOURCES})
    get_filename_component(shader_ext ${shader} EXT)
    string(SUBSTRING ${shader_ext} 1 -1 shader_type)
    set(output_file ${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader_type}.spv)

    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${GLSL_COMPILER} ${CMAKE_CURRENT_SOURCE_DIR}/${shader} -o ${output_file}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${shader}
        COMMENT "Compiling ${shader} to ${shader_type}.spv"
    )

    list(APPEND SHADER_SPV_FILES ${output_file})
endforeach()

add_custom_target(compile_shaders_vertex_buffers ALL
    DEPENDS ${SHADER_SPV_FILES}
    COMMENT "Building all shaders"
)

# executable
add_executable(main_vertex_buffers main.cpp)
add_dependencies(main_vertex_buffers compile_shaders_vertex_buffers)
target_link_libraries(main_vertex_buffers PRIVATE
    glm::glm
    glfw
    ${Vulkan_LIBRARIES}
)
target_include_directories(main_vertex_buffers PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/shaders
)