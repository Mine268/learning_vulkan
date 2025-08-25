cmake_minimum_required(VERSION 3.12)

project(5_depth_buffering)

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

add_custom_target(compile_depth_buffering ALL
    DEPENDS ${SHADER_SPV_FILES}
    COMMENT "Building all shaders"
)

# executable
add_executable(main_depth_buffering main.cpp)
add_dependencies(main_depth_buffering compile_depth_buffering)
target_link_libraries(main_depth_buffering PRIVATE
    glm::glm
    glfw
    ${Vulkan_LIBRARIES}
)
target_include_directories(main_depth_buffering PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/shaders
)