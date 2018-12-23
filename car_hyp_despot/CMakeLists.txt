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


project(car_hyp_despot CXX CUDA)


if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()


# set(CUDAHOSTCXX "g++-6")

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

# message("==============================CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}" )


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
set(CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH} "${PROJECT_SOURCE_DIR}/cmake")


set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#set (CMAKE_CXX_FLAGS "--std=c++11")
#set (CUDA_PROPAGATE_HOST_FLAGS ON)
#set (CUDA_64_BIT_DEVICE_CODE ON)

if (CUDA)
  #set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "Compile CUDA objects with separable compilation enabled.")
  #set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL "Use the static version of the CUDA runtime library if available")
endif()


# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_30,code=sm_30
    #-g -G
    )

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  message_generation
)

set(CMAKE_PREFIX_PATH  "${CMAKE_PREFIX_PATH};/opt/libtorch")
# set(CUDA_HOST_COMPILER "/usr/bin/gcc-5")
# set(CUDAHOSTCXX "g++-5")
# set(CMAKE_C_COMPILER "/usr/bin/gcc-5")

# set(CMAKE_CXX_COMPILER "/usr/bin/gcc-5")

# set(CMAKE_C_COMPILER_VERSION 5.5)


find_package(Torch REQUIRED)

find_package(OpenCV 4.0.0 REQUIRED)

#add_subdirectory(src/HypDespot/)

add_message_files(
    DIRECTORY msg
    FILES
    ped_belief.msg
    peds_believes.msg
)
generate_messages(
    DEPENDENCIES
        std_msgs
        geometry_msgs
        # ped_is_despot
        car_hyp_despot
        query_nn
)

include_directories(
   ${catkin_INCLUDE_DIRS}
   ${OpenCV_INCLUDE_DIRS}
   src/RVO2/src
   src/HypDespot/include/
   src/HypDespot/include/despot/
   src/planner/
   src/planner/GPU_Car_Drive/
   ../ped_is_despot/src/

)

FILE(GLOB LIBRVO_SRCS  src/RVO2/src/*.cpp)

catkin_package(
  INCLUDE_DIRS src/planner/
#  LIBRARIES ped_path_predictor
  CATKIN_DEPENDS roscpp query_nn #std_msgs
  DEPENDS system_lib
)
message("@@@@@@@@@@@@@@ before add lib @@@@@@@@@@@@@@")

#add_subdirectory(src/HypDespot/)
#add_subdirectory(src/planner)

#set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} dummmmmy")

message("==============================CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

string(REPLACE " dummmmmy" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
message("==============================CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")


add_library(
"${PROJECT_NAME}" 
# SHARED
STATIC

  src/HypDespot/src/interface/pomdp.cpp
  src/HypDespot/src/interface/world.cpp
  src/HypDespot/src/interface/belief.cpp
  src/HypDespot/src/interface/lower_bound.cpp
  src/HypDespot/src/interface/upper_bound.cpp
  src/HypDespot/src/interface/default_policy.cpp
  src/HypDespot/src/interface/policy_graph.cpp
  src/HypDespot/src/core/particle_belief.cpp
  src/HypDespot/src/core/globals.cpp
  src/HypDespot/src/core/builtin_lower_bounds.cpp
  src/HypDespot/src/core/mdp.cpp
  src/HypDespot/src/core/node.cpp
  src/HypDespot/src/core/builtin_policy.cpp
  src/HypDespot/src/core/builtin_policygraph.cpp
  src/HypDespot/src/core/pomdp_world.cpp
  src/HypDespot/src/core/solver.cpp
  src/HypDespot/src/core/builtin_upper_bounds.cpp
  src/HypDespot/src/logger.cpp
  src/HypDespot/src/planner.cpp
  src/HypDespot/src/Parallel_planner.cu
  src/HypDespot/src/evaluator.cpp
  src/HypDespot/src/pomdpx/parser/function.cpp
  src/HypDespot/src/pomdpx/parser/parser.cpp
  src/HypDespot/src/pomdpx/parser/variable.cpp
  src/HypDespot/src/pomdpx/pomdpx.cpp
  src/HypDespot/src/random_streams.cpp
  src/HypDespot/src/plannerbase.cpp
  src/HypDespot/src/solver/aems.cpp
  src/HypDespot/src/solver/despot.cpp
  src/HypDespot/src/solver/pomcp.cpp
  src/HypDespot/src/solver/baseline_solver.cpp
  src/HypDespot/src/util/coord.cpp
  src/HypDespot/src/util/dirichlet.cpp
  src/HypDespot/src/util/exec_tracker.cpp
  src/HypDespot/src/util/floor.cpp
  src/HypDespot/src/util/gamma.cpp
  src/HypDespot/src/util/logging.cpp
  src/HypDespot/src/util/random.cpp
  src/HypDespot/src/util/seeds.cpp
  src/HypDespot/src/util/util.cpp
  src/HypDespot/src/util/error_handler.cpp
  src/HypDespot/src/util/tinyxml/tinystr.cpp
  src/HypDespot/src/util/tinyxml/tinyxml.cpp
  src/HypDespot/src/util/tinyxml/tinyxmlerror.cpp
  src/HypDespot/src/util/tinyxml/tinyxmlparser.cpp

  src/HypDespot/src/GPUinterface/GPUdefault_policy.cu  
  src/HypDespot/src/GPUinterface/GPUpomdp.cu 
  src/HypDespot/src/GPUinterface/GPUlower_bound.cu  
  src/HypDespot/src/GPUinterface/GPUupper_bound.cu  
  src/HypDespot/src/GPUcore/GPUbuiltin_lower_bound.cu
  src/HypDespot/src/GPUcore/GPUbuiltin_policy.cu
  src/HypDespot/src/GPUcore/GPUpolicy_graph.cu
  src/HypDespot/src/GPUcore/GPUbuiltin_upper_bound.cu
  src/HypDespot/src/GPUcore/GPUconfig.cu   
  src/HypDespot/src/GPUcore/GPUhistory.cu      
  src/HypDespot/src/GPUcore/GPUnode.cu  
  src/HypDespot/src/GPUcore/GPUglobals.cu  
  src/HypDespot/src/GPUcore/shared_node.cu
  src/HypDespot/src/GPUcore/thread_globals.cu
  src/HypDespot/src/GPUutil/GPUmemorypool.cu  
  src/HypDespot/src/GPUutil/GPUseeds.cu    
  src/HypDespot/src/GPUutil/GPUutil.cu
  src/HypDespot/src/GPUutil/GPUcoord.cu      
  src/HypDespot/src/GPUutil/GPUrandom.cu      
  src/HypDespot/src/GPUrandom_streams.cu
  src/HypDespot/src/solver/Hyp_despot.cu

  ${LIBRVO_SRCS}

  src/planner/Path.cpp
  src/planner/collision.cpp
  src/planner/coord.cpp                  
  src/planner/math_utils.cpp
  src/planner/ped_pomdp.cpp
  src/planner/neural_prior.cpp
  src/planner/WorldModel.cpp
  src/planner/custom_particle_belief.cpp  
  src/planner/param.cpp
  # src/planner/simulator.cpp
  # src/planner/controller.cpp
  src/planner/GPU_Car_Drive/GPU_Car_Drive.cu     
  src/planner/GPU_Car_Drive/GPU_LowerBoundPolicy.cu
  src/planner/GPU_Car_Drive/GPU_Init.cu         
  src/planner/GPU_Car_Drive/GPU_Path.cu
  src/planner/GPU_Car_Drive/GPU_CarUpperBound.cu  
  src/planner/GPU_Car_Drive/GPU_param.cu  



  
  # src/HypDespot/src/core/belief.cpp
  # src/HypDespot/src/core/globals.cpp
  # src/HypDespot/src/core/lower_bound.cpp
  # src/HypDespot/src/core/mdp.cpp
  # src/HypDespot/src/core/node.cpp
  # src/HypDespot/src/core/policy.cpp
  # src/HypDespot/src/core/policy_graph.cpp
  # src/HypDespot/src/core/pomdp.cpp
  # src/HypDespot/src/core/solver.cpp
  # src/HypDespot/src/core/history.cpp
  # src/HypDespot/src/core/upper_bound.cpp
  # src/HypDespot/src/evaluator.cpp
  # src/HypDespot/src/ippc/client.cpp
  # src/HypDespot/src/pomdpx/parser/function.cpp
  # src/HypDespot/src/pomdpx/parser/parser.cpp
  # src/HypDespot/src/pomdpx/parser/variable.cpp
  # src/HypDespot/src/pomdpx/pomdpx.cpp
  # src/HypDespot/src/random_streams.cpp
  # src/HypDespot/src/simple_tui.cpp
  # src/HypDespot/src/solver/aems.cpp
  # src/HypDespot/src/solver/despot.cpp
  # src/HypDespot/src/solver/pomcp.cpp
  # src/HypDespot/src/util/coord.cpp
  # src/HypDespot/src/util/dirichlet.cpp
  # src/HypDespot/src/util/exec_tracker.cpp
  # src/HypDespot/src/util/floor.cpp
  # src/HypDespot/src/util/gamma.cpp
  # src/HypDespot/src/util/logging.cpp
  # src/HypDespot/src/util/random.cpp
  # src/HypDespot/src/util/seeds.cpp
  # src/HypDespot/src/util/util.cpp
  # src/HypDespot/src/util/tinyxml/tinystr.cpp
  # src/HypDespot/src/util/tinyxml/tinyxml.cpp
  # src/HypDespot/src/util/tinyxml/tinyxmlerror.cpp
  # src/HypDespot/src/util/tinyxml/tinyxmlparser.cpp
  # src/HypDespot/src/GPUcore/GPUbelief.cu   
  # src/HypDespot/src/GPUcore/GPUconfig.cu   
  # src/HypDespot/src/GPUcore/GPUhistory.cu      
  # src/HypDespot/src/GPUcore/GPUnode.cu  
  # src/HypDespot/src/GPUcore/GPUpolicy.cu        
  # src/HypDespot/src/GPUcore/GPUpomdp.cu  
  # src/HypDespot/src/GPUcore/shared_node.cu
  # src/HypDespot/src/GPUcore/GPUglobals.cu  
  # src/HypDespot/src/GPUcore/GPUlower_bound.cu  
  # src/HypDespot/src/GPUcore/GPUpolicy_graph.cu  
  # src/HypDespot/src/GPUcore/thread_globals.cu
  # src/HypDespot/src/GPUutil/GPUcstdlib.cu  
  # src/HypDespot/src/GPUutil/GPUdebug.cu  
  # src/HypDespot/src/GPUutil/GPUHostReflection.cu  
  # src/HypDespot/src/GPUutil/GPUmemorypool.cu  
  # src/HypDespot/src/GPUutil/GPUseeds.cu    
  # src/HypDespot/src/GPUutil/GPUutil.cu
  # src/HypDespot/src/GPUutil/GPUcoord.cu      
  # src/HypDespot/src/GPUutil/GPUcstring.cu  
  # src/HypDespot/src/GPUutil/GPUFile.cu   
  # src/HypDespot/src/GPUutil/GPUKnob.cu            
  # src/HypDespot/src/GPUutil/GPUrandom.cu      
  # src/HypDespot/src/GPUutil/GPUThreadId.cu 
  # src/HypDespot/src/GPUrandom_streams.cu
  # src/HypDespot/src/solver/GPUdespot.cu
  # ${LIBRVO_SRCS}
  # src/planner/Path.cpp
  # src/planner/simulator.cu
  # src/planner/collision.cpp
  # src/planner/coord.cpp                  
  # src/planner/math_utils.cpp
  # src/planner/ped_pomdp.cpp
  # src/planner/WorldModel.cpp
  # src/planner/custom_particle_belief.cpp  
  # src/planner/param.cpp
  # src/planner/GPU_Car_Drive/GPU_Car_Drive.cu     
  # src/planner/GPU_Car_Drive/GPU_LowerBoundPolicy.cu
  # src/planner/GPU_Car_Drive/GPU_Init.cu         
  # src/planner/GPU_Car_Drive/GPU_Path.cu
  # src/planner/GPU_Car_Drive/GPU_CarUpperBound.cu  
  # src/planner/GPU_Car_Drive/GPU_param.cu  

 # src/dummy.cpp
)
message("@@@@@@@@@@@@@@ after add lib @@@@@@@@@@@@@@")


add_dependencies("${PROJECT_NAME}" ped_is_despot_generate_messages_cpp)
add_dependencies("${PROJECT_NAME}" query_nn_generate_messages_cpp)

# add_dependencies("${PROJECT_NAME}" car_hyp_despot_generate_messages_cpp)

#add_compile_options(${PROJECT_NAME}
#    PRIVATE
#    $<$<COMPILE_LANGUAGE:CUDA>:-arch=compute_30,code=sm_30>
#)

set_target_properties( "${PROJECT_NAME}"
			#GPUmodel
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties( "${PROJECT_NAME}"
			#GPUmodel
                       PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(${PROJECT_NAME}
  #GPUmodel  
  PRIVATE KW_EXPORTS
  INTERFACE KW_IMPORTS
)

target_link_libraries("${PROJECT_NAME}"
  #GPUmodel
  # ${catkin_LIBRARIES}
  ${TinyXML_LIBRARIES}
  #HypDespot
)

message("Torch library list:  ${TORCH_LIBRARIES}")
target_link_libraries("${PROJECT_NAME}" "${TORCH_LIBRARIES}")

target_link_libraries("${PROJECT_NAME}" "${OpenCV_LIBRARIES}")

# target_link_libraries("${PROJECT_NAME}" 
#   torch
#   caffe2_library
#   caffe2_gpu_library
#   )




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

