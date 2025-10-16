


set(LIBRARY_DATASET_FOR_TEST_SOURCES
    ../src/datasets/botanic_gardent_dataset.cc
    ../src/datasets/fast_livo2_dataset.cc
    ../src/datasets/remode_dataset.cc
    ../src/datasets/tartan_air_dataset.cc
    ../src/datasets/kitti_dataset.cc
    ../src/datasets/virtual_kitti_dataset.cc
    ../src/datasets/dataset.cc
)

add_executable(
    run_visualize_trajectories
    visualize_trajectories.cc
    ../src/configs.cc
    ../src/undistort.cc
    ../src/utils.cc
    ../src/visualizer.cc
    ${LIBRARY_DATASET_FOR_TEST_SOURCES}
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
    run_visualize_colorized_gtmap
    visualize_colorized_gtmap.cc
    ../src/configs.cc
    ../src/undistort.cc
    ../src/utils.cc
    ../src/visualizer.cc
    ${LIBRARY_DATASET_FOR_TEST_SOURCES}
)

target_include_directories(run_visualize_colorized_gtmap
    PUBLIC
    ${DIRS_HEADER_THIRDPARTY}
)

target_link_libraries(run_visualize_colorized_gtmap
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

