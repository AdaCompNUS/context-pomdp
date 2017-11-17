cmake_minimum_required(VERSION 3.8)
message("@@@@@@@@@@@@@@ HypDespot cmake list @@@@@@@@@@@@@@")
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


#project(HypDespot CXX CUDA)


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
set(CONFIG_INSTALL_PATH "${LIBRARY_INSTALL_PATH}/HypDespot/cmake")

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
   include
   ../planner/
   ../planner/GPU_Car_Drive/
)

#set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

add_library(#"${PROJECT_NAME}" STATIC
"HypDespot" STATIC
#add_executable(${PROJECT_NAME}  
  src/core/belief.cpp
  src/core/globals.cpp
  src/core/lower_bound.cpp
  src/core/mdp.cpp
  src/core/node.cpp
  src/core/policy.cpp
  src/core/policy_graph.cpp
  src/core/pomdp.cpp
  src/core/solver.cpp
  src/core/history.cpp
  src/core/upper_bound.cpp
  src/evaluator.cpp
  src/ippc/client.cpp
  src/pomdpx/parser/function.cpp
  src/pomdpx/parser/parser.cpp
  src/pomdpx/parser/variable.cpp
  src/pomdpx/pomdpx.cpp
  src/random_streams.cpp
  src/simple_tui.cpp
  src/solver/aems.cpp
  src/solver/despot.cpp
  src/solver/pomcp.cpp
  src/util/coord.cpp
  src/util/dirichlet.cpp
  src/util/exec_tracker.cpp
  src/util/floor.cpp
  src/util/gamma.cpp
  src/util/logging.cpp
  src/util/random.cpp
  src/util/seeds.cpp
  src/util/util.cpp
  src/util/tinyxml/tinystr.cpp
  src/util/tinyxml/tinyxml.cpp
  src/util/tinyxml/tinyxmlerror.cpp
  src/util/tinyxml/tinyxmlparser.cpp
  src/GPUcore/GPUbelief.cu   
  src/GPUcore/GPUconfig.cu   
  src/GPUcore/GPUhistory.cu      
  src/GPUcore/GPUnode.cu  
  src/GPUcore/GPUpolicy.cu        
  src/GPUcore/GPUpomdp.cu  
  src/GPUcore/shared_node.cu
  src/GPUcore/GPUglobals.cu  
  src/GPUcore/GPUlower_bound.cu  
  src/GPUcore/GPUpolicy_graph.cu  
  src/GPUcore/thread_globals.cu
  src/GPUutil/GPUcstdlib.cu  
  src/GPUutil/GPUdebug.cu  
  src/GPUutil/GPUHostReflection.cu  
  src/GPUutil/GPUmemorypool.cu  
  src/GPUutil/GPUseeds.cu    
  src/GPUutil/GPUutil.cu
  src/GPUutil/GPUcoord.cu      
  src/GPUutil/GPUcstring.cu  
  src/GPUutil/GPUFile.cu   
  src/GPUutil/GPUKnob.cu            
  src/GPUutil/GPUrandom.cu      
  src/GPUutil/GPUThreadId.cu 
  src/GPUrandom_streams.cu
  src/solver/GPUdespot.cu
  #OPTIONS --compiler-options "-fPIC"
)

#add_compile_options(${PROJECT_NAME}
#    PRIVATE
#    $<$<COMPILE_LANGUAGE:CUDA>:-arch=compute_30,code=sm_30>
#)

set_target_properties( "HypDespot"
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties( "HypDespot"
                       PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(HypDespot
  PRIVATE KW_EXPORTS
  INTERFACE KW_IMPORTS
)

target_link_libraries("HypDespot"
  ${TinyXML_LIBRARIES}
)



install(TARGETS "HypDespot"
  EXPORT "DespotTargets"
  ARCHIVE DESTINATION "${LIBRARY_INSTALL_PATH}"
  LIBRARY DESTINATION "${LIBRARY_INSTALL_PATH}"
  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
)
install(DIRECTORY "include/HypDespot/"
  DESTINATION "${INCLUDE_INSTALL_PATH}/HypDespot"
)

# Install a DespotConfig.cmake file so CMake can find_package(Despot).
include(CMakePackageConfigHelpers)
configure_package_config_file("cmake/DespotConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/DespotConfig.cmake"
  INSTALL_DESTINATION "${CONFIG_INSTALL_PATH}"
  PATH_VARS INCLUDE_INSTALL_PATH
)

install(EXPORT "DespotTargets"
  FILE "DespotTargets.cmake"
  DESTINATION "${CONFIG_INSTALL_PATH}"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DespotConfig.cmake"
  DESTINATION "${CONFIG_INSTALL_PATH}"
)

message("@@@@@@@@@@@@@@ HypDespot end @@@@@@@@@@@@@@")

