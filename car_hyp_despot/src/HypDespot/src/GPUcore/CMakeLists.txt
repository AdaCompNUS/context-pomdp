cmake_minimum_required(VERSION 2.8.3)

#Find cuda package
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_30,code=sm_30
)

# Specify include directories
include_directories(
    kernels
    utility
)
cuda_add_library("${PROJECT_NAME}_GPU" SHARED
  #GPUbelief.cu
  #GPUglobals.cu
  #GPUlower_bound.cu
  #GPUmdp.cu
  #GPUnode.cu
  #GPUpolicy.cu
  #GPUpomdp.cu
  #GPUsolver.cu
  #GPUupper_bound.cu
  ../GPUutil/GPUcoord.cu
  ../GPUutil/GPUcstdlib.cu
  ../GPUutil/GPUcstring.cu
  ../GPUutil/GPUdebug.cu
  ../GPUutil/GPUFile.cu
  ../GPUutil/GPUHostReflection.cu
  ../GPUutil/GPUKnob.cu
  ../GPUutil/GPUThreadId.cu
  ../GPUutil/GPUvector.inl
  ../GPUrandom_streams.cu
  GPUnode.cu
  GPUpomdp.cu
  GPU_Unk_nav/GPU_base_unc_navigation.cu
  GPU_Unk_nav/GPU_UncNavigation.cu
  ../solver/GPUdespot.cu
  #../GPUutil/GPUseeds.cu
  #../GPUutil/GPUrandom.cu
)

target_link_libraries("${PROJECT_NAME}_GPU"
    ${PROJECT_NAME}
    )

install(TARGETS "${PROJECT_NAME}_GPU"
  EXPORT "DespotGPUTargets"
  ARCHIVE DESTINATION "${LIBRARY_INSTALL_PATH}"
  LIBRARY DESTINATION "${LIBRARY_INSTALL_PATH}"
  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
)

