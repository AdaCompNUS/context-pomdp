cmake_minimum_required(VERSION 3.8)
message("@@@@@@@@@@@@@@ Hyp-despot cmake list @@@@@@@@@@@@@@")

set(CUDA_SDK_ROOT_DIR "/usr/local/cuda/samples")

#project(HyP_DESPOT_Hyp-despot CXX CUDA)


if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED True)
endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED True)
endif()


# set(BINARY_INSTALL_PATH "bin" CACHE PATH "Binary install path")
# set(LIBRARY_INSTALL_PATH "lib" CACHE PATH "Library install path")
# set(INCLUDE_INSTALL_PATH "include" CACHE PATH "Include install path")
# set(CONFIG_INSTALL_PATH "${LIBRARY_INSTALL_PATH}/Hyp-despot/cmake")

# set(DESPOT_BUILD_EXAMPLES ON CACHE BOOL "Build C++ model examples")
# set(DESPOT_BUILD_POMDPX ON CACHE BOOL "Build POMDPX example")
# set(DESPOT_BUILD_GPUS ON CACHE BOOL "Build GPU despot")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30")
# set(CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH} "${PROJECT_SOURCE_DIR}/cmake")

if (CUDA)
endif()

#add_definitions(-std=c++0x -O3)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lunwind")


find_package(catkin REQUIRED COMPONENTS
  roscpp
#  ${PROJECT_NAME}
)


include_directories(
   ${catkin_INCLUDE_DIRS}
   include/
)

message("@@@@@@@@@@@@@@ before add lib @@@@@@@@@@@@@@")

add_library(
# "${PROJECT_NAME}" STATIC
"HypDespot" STATIC

  src/interface/pomdp.cpp
  src/interface/world.cpp
  src/interface/belief.cpp
  src/interface/lower_bound.cpp
  src/interface/upper_bound.cpp
  src/interface/default_policy.cpp
  src/interface/policy_graph.cpp
  src/core/particle_belief.cpp
  src/core/globals.cpp
  src/core/builtin_lower_bounds.cpp
  src/core/mdp.cpp
  src/core/node.cpp
  src/core/builtin_policy.cpp
  src/core/builtin_policygraph.cpp
  src/core/pomdp_world.cpp
  src/core/solver.cpp
  src/core/builtin_upper_bounds.cpp
  src/logger.cpp
  src/planner.cpp
  src/Parallel_planner.cu
  src/evaluator.cpp
  src/pomdpx/parser/function.cpp
  src/pomdpx/parser/parser.cpp
  src/pomdpx/parser/variable.cpp
  src/pomdpx/pomdpx.cpp
  src/random_streams.cpp
  src/plannerbase.cpp
  src/solver/aems.cpp
  src/solver/despot.cpp
  src/solver/pomcp.cpp
  src/solver/baseline_solver.cpp
  src/util/coord.cpp
  src/util/dirichlet.cpp
  src/util/exec_tracker.cpp
  src/util/floor.cpp
  src/util/gamma.cpp
  src/util/logging.cpp
  src/util/random.cpp
  src/util/seeds.cpp
  src/util/util.cpp
  src/util/error_handler.cpp
  src/util/tinyxml/tinystr.cpp
  src/util/tinyxml/tinyxml.cpp
  src/util/tinyxml/tinyxmlerror.cpp
  src/util/tinyxml/tinyxmlparser.cpp

  src/GPUinterface/GPUdefault_policy.cu  
  src/GPUinterface/GPUpomdp.cu 
  src/GPUinterface/GPUlower_bound.cu  
  src/GPUinterface/GPUupper_bound.cu  
  src/GPUcore/GPUbuiltin_lower_bound.cu
  src/GPUcore/GPUbuiltin_policy.cu
  src/GPUcore/GPUpolicy_graph.cu
  src/GPUcore/GPUbuiltin_upper_bound.cu
  src/GPUcore/GPUconfig.cu   
  src/GPUcore/GPUhistory.cu      
  src/GPUcore/GPUnode.cu  
  src/GPUcore/GPUglobals.cu  
  src/GPUcore/shared_node.cu
  src/GPUcore/thread_globals.cu
  src/GPUutil/GPUmemorypool.cu  
  src/GPUutil/GPUseeds.cu    
  src/GPUutil/GPUutil.cu
  src/GPUutil/GPUcoord.cu      
  src/GPUutil/GPUrandom.cu      
  src/GPUrandom_streams.cu
  src/solver/Hyp_despot.cu
  ${LIBRVO_SRCS}
  # src/planner/Path.cpp
  # src/planner/simulator.cu
  #   src/planner/collision.cpp
  #   src/planner/coord.cpp                  
  #   src/planner/math_utils.cpp
  #   src/planner/ped_pomdp.cpp
  #   src/planner/WorldModel.cpp
  #   src/planner/custom_particle_belief.cpp  
  #   src/planner/param.cpp
  #   src/planner/GPU_Car_Drive/GPU_Car_Drive.cu     
  #   src/planner/GPU_Car_Drive/GPU_LowerBoundPolicy.cu
  #   src/planner/GPU_Car_Drive/GPU_Init.cu         
  #   src/planner/GPU_Car_Drive/GPU_Path.cu
  #   src/planner/GPU_Car_Drive/GPU_CarUpperBound.cu  
  #   src/planner/GPU_Car_Drive/GPU_param.cu  
  #OPTIONS --compiler-options "-fPIC"
)
message("@@@@@@@@@@@@@@ after add lib @@@@@@@@@@@@@@")

set_target_properties( "HypDespot"
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties( "HypDespot"
                       PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_definitions(HypDespot
  PRIVATE KW_EXPORTS
  INTERFACE KW_IMPORTS
)

message("@@@@@@@@@@@@@@ LIBUNWIND_LIBRARIES @@@@@@@@@@@@@@")

target_link_libraries(HypDespot # "${PROJECT_NAME}"
   ${LIBUNWIND_LIBRARIES}
 )
message("@@@@@@@@@@@@@@ end @@@@@@@@@@@@@@")

