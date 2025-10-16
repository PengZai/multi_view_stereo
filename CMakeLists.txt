cmake_minimum_required(VERSION 3.3)
project(multi_view_stereo)


# find package
find_package(OpenCV 4 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Matplot++ REQUIRED)
find_package(Pangolin 0.6 REQUIRED)
find_package(PCL REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Setting CMAKE_CXX_EXTENSIONS to OFF ensures strict compliance with the selected C++ standard.


option(ENABLE_CMAKE_DEBUG "Enable detailed debug messages" OFF)

if(ENABLE_CMAKE_DEBUG)
    # Debug or Release
    set(CMAKE_BUILD_TYPE "Debug")
else()
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Enable compile optimizations
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
# Enable debug flags (use if you want to debug in gdb)
set(CMAKE_CXX_FLAGS_DEBUG "-g3 -fno-omit-frame-pointer")



list(APPEND LIBRARY_THIRDPARTY
    ${OpenCV_LIBS}
    ${Pangolin_LIBRARIES}
    Matplot++::matplot
    ${Pangolin_LIBRARIES}
    ${PCL_LIBRARIES}
)


list(APPEND DIRS_HEADER_THIRDPARTY
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)


list(APPEND DIRS_HEADER_LIBRARY

    src
    src/datasets
)

add_definitions(${PCL_DEFINITIONS})


set(LIBRARY_DATASET_SOURCES
    src/datasets/botanic_gardent_dataset.cc
    src/datasets/fast_livo2_dataset.cc
    src/datasets/remode_dataset.cc
    src/datasets/tartan_air_dataset.cc
    src/datasets/kitti_dataset.cc
    src/datasets/virtual_kitti_dataset.cc
    src/datasets/dataset.cc
)

set(LIBRARY_SOURCES
    src/configs.cc
    src/multi_view_stereo.cc
    src/utils.cc
    src/undistort.cc
    src/visualizer.cc
    
)

add_executable(
    run_multi_view_stereo
    src/main.cc
    ${LIBRARY_SOURCES}
    ${LIBRARY_DATASET_SOURCES}
)

target_include_directories(run_multi_view_stereo
    PUBLIC
    ${DIRS_HEADER_LIBRARY}
    ${DIRS_HEADER_THIRDPARTY}
)

target_link_libraries(run_multi_view_stereo
    PUBLIC
    ${LIBRARY_THIRDPARTY}
)



add_subdirectory(test)
