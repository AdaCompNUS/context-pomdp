cmake_minimum_required(VERSION 3.8)

set(CUDA_SDK_ROOT_DIR "/usr/local/cuda/samples")

#project(HyP_DESPOT_CarDriving CXX CUDA)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lunwind")

# set(BINARY_INSTALL_PATH "bin" CACHE PATH "Binary install path")
# set(LIBRARY_INSTALL_PATH "lib" CACHE PATH "Library install path")
# set(INCLUDE_INSTALL_PATH "include" CACHE PATH "Include install path")
# set(CONFIG_INSTALL_PATH "${LIBRARY_INSTALL_PATH}/CarDriving/cmake")

# set(DESPOT_BUILD_EXAMPLES ON CACHE BOOL "Build C++ model examples")
# set(DESPOT_BUILD_POMDPX ON CACHE BOOL "Build POMDPX example")
# set(DESPOT_BUILD_GPUS ON CACHE BOOL "Build GPU despot")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
# set(CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH} "${PROJECT_SOURCE_DIR}/cmake")

if (CUDA)
endif()

#add_definitions(-std=c++0x -O3)

find_package(catkin REQUIRED COMPONENTS
  roscpp
#  ${PROJECT_NAME}
)


include_directories(
   ${catkin_INCLUDE_DIRS}
   ../../HypDespot/include/
   ./
)

add_library(
#"${PROJECT_NAME}" STATIC
"GPUmodel" STATIC
# add_executable(${PROJECT_NAME}_CarDriving  
  path.cpp
  collision.cpp
  coord.cpp                  
  math_utils.cpp
  ped_pomdp.cpp
  world_model.cpp
  custom_particle_belief.cpp  
  param.cpp
  simulator.cpp
  controller.cpp
)

target_link_libraries(GPUmodel # ${PROJECT_NAME}_CarDriving
   ${catkin_LIBRARIES}
   ${PROJECT_NAME}
)

set_target_properties( 
			GPUmodel # ${PROJECT_NAME}_CarDriving
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties( 
			GPUmodel # ${PROJECT_NAME}_CarDriving
                       PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(GPUmodel # ${PROJECT_NAME}_CarDriving
  PRIVATE KW_EXPORTS
  INTERFACE KW_IMPORTS
)

target_link_libraries( GPUmodel # ${PROJECT_NAME}_CarDriving
  ${TinyXML_LIBRARIES}
)

target_link_libraries(GPUmodel # ${PROJECT_NAME}_CarDriving
   ${LIBUNWIND_LIBRARIES}
 )

# Install a DespotConfig.cmake file so CMake can find_package(Despot).
include(CMakePackageConfigHelpers)
configure_package_config_file("cmake/DespotConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/DespotConfig.cmake"
  INSTALL_DESTINATION "${CONFIG_INSTALL_PATH}"
  PATH_VARS INCLUDE_INSTALL_PATH
)



