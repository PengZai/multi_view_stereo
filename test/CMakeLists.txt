



add_executable(
    run_visualize_trajectories
    visualize_trajectories.cc
    ../src/configs.cc
    ../src/data.cc
    ../src/utils.cc
)

target_include_directories(run_visualize_trajectories
    PUBLIC
    ${DIRS_HEADER_THIRDPARTY}
)

target_link_libraries(run_visualize_trajectories
    PUBLIC
    ${LIBRARY_THIRDPARTY}
)


add_executable(
    run_pixel_operation_comparison
    pixel_operation_comparison.cc
)

target_include_directories(run_pixel_operation_comparison
    PUBLIC
    ${DIRS_HEADER_THIRDPARTY}
)

target_link_libraries(run_pixel_operation_comparison
    PUBLIC
    ${LIBRARY_THIRDPARTY}
)

