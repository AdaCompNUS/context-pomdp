cmake_minimum_required(VERSION 3.8)
message("@@@@@@@@@@@@@@ GPUmodel cmake list @@@@@@@@@@@@@@")
#SET(CMAKE_MODULE_PATH
#  "${CMAKE_SOURCE_DIR}/cmake"
#"${CMAKE_SOURCE_DIR}/CMake/cuda" 
#  "${CMAKE_MODULE_PATH}")
#SET(CMAKE_MODULE_PATH "/usr/local/share/cmake/torch" ${CMAKE_MODULE_PATH})
#find_package(CUDA REQUIRED)
#message("after find cuda")

#project(GPUDespot LANGUAGES CXX CUDA)
#set(CUDAHOSTCXX "g++-4")
#set(CMAKE_CUDA_COMPILER "/usr/local/cuda-8.0/bin/nvcc")
set(CUDA_SDK_ROOT_DIR "/usr/local/cuda/samples")


#project(GPUmodel CXX CUDA)


if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()
#find_package(CUDA QUIET REQUIRED)


#set(CUDA_SDK_ROOT_DIR "/usr/local/cuda/samples")
#set(CUDA_CUDART_LIBRARY "/usr/local/cuda/lib64/libcudart.so ../")


set(BINARY_INSTALL_PATH "bin" CACHE PATH "Binary install path")
set(LIBRARY_INSTALL_PATH "lib" CACHE PATH "Library install path")
set(INCLUDE_INSTALL_PATH "include" CACHE PATH "Include install path")
set(CONFIG_INSTALL_PATH "${LIBRARY_INSTALL_PATH}/GPUmodel/cmake")

set(DESPOT_BUILD_EXAMPLES ON CACHE BOOL "Build C++ model examples")
set(DESPOT_BUILD_POMDPX ON CACHE BOOL "Build POMDPX example")
set(DESPOT_BUILD_GPUS ON CACHE BOOL "Build GPU despot")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
set(CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH} "${PROJECT_SOURCE_DIR}/cmake")

#set (CMAKE_CXX_FLAGS "--std=c++11")
#set (CUDA_PROPAGATE_HOST_FLAGS ON)
#set (CUDA_64_BIT_DEVICE_CODE ON)

if (CUDA)
  #set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "Compile CUDA objects with separable compilation enabled.")
  #set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "Use the static version of the CUDA runtime library if available")
endif()


# Pass options to NVCC
#set(
#    CUDA_NVCC_FLAGS
#    ${CUDA_NVCC_FLAGS};
#    -O3 -gencode arch=compute_30,code=sm_30
#    )

find_package(catkin REQUIRED COMPONENTS
  roscpp
)


include_directories(
   ${catkin_INCLUDE_DIRS}
   ../HypDespot/include/
   ./
   GPU_Car_Drive/
)
message("@@@@@@@@@@@@@@ before add lib @@@@@@@@@@@@@@")

#set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

add_library(
#"${PROJECT_NAME}" STATIC
"GPUmodel" STATIC
#add_executable(${PROJECT_NAME}  
  Path.cpp
  #simulator.cu
  collision.cpp
  coord.cpp                  
  math_utils.cpp
  ped_pomdp.cpp
  WorldModel.cpp
  custom_particle_belief.cpp  
  param.cpp
  simulator.cu
  GPU_Car_Drive/GPU_Car_Drive.cu     
  GPU_Car_Drive/GPU_LowerBoundPolicy.cu
  GPU_Car_Drive/GPU_Init.cu         
  GPU_Car_Drive/GPU_Path.cu
  GPU_Car_Drive/GPU_CarUpperBound.cu  
  GPU_Car_Drive/GPU_param.cu  
  #OPTIONS --compiler-options "-fPIC"
)
message("@@@@@@@@@@@@@@ after add lib @@@@@@@@@@@@@@")

#add_compile_options(${PROJECT_NAME}
#    PRIVATE
#    $<$<COMPILE_LANGUAGE:CUDA>:-arch=compute_30,code=sm_30>
#)

set_target_properties( #"${PROJECT_NAME}"
			GPUmodel
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties( #"${PROJECT_NAME}"
			GPUmodel
                       PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(#${PROJECT_NAME}
  GPUmodel  
  PRIVATE KW_EXPORTS
  INTERFACE KW_IMPORTS
)

target_link_libraries(#"${PROJECT_NAME}"
  GPUmodel
  ${TinyXML_LIBRARIES}
)



#install(TARGETS "${PROJECT_NAME}"
#  EXPORT "DespotTargets"
#  ARCHIVE DESTINATION "${LIBRARY_INSTALL_PATH}"
#  LIBRARY DESTINATION "${LIBRARY_INSTALL_PATH}"
#  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
#)
#install(DIRECTORY "include/${PROJECT_NAME}/"
#  DESTINATION "${INCLUDE_INSTALL_PATH}/${PROJECT_NAME}"
#)

# Install a DespotConfig.cmake file so CMake can find_package(Despot).
include(CMakePackageConfigHelpers)
configure_package_config_file("cmake/DespotConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/DespotConfig.cmake"
  INSTALL_DESTINATION "${CONFIG_INSTALL_PATH}"
  PATH_VARS INCLUDE_INSTALL_PATH
)

#install(EXPORT "DespotTargets"
#  FILE "DespotTargets.cmake"
#  DESTINATION "${CONFIG_INSTALL_PATH}"
#)
#install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DespotConfig.cmake"
#  DESTINATION "${CONFIG_INSTALL_PATH}"
#)

message("@@@@@@@@@@@@@@ end @@@@@@@@@@@@@@")

