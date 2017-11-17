#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "GPUmodel" for configuration ""
set_property(TARGET GPUmodel APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(GPUmodel PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CUDA;CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libGPUmodel.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS GPUmodel )
list(APPEND _IMPORT_CHECK_FILES_FOR_GPUmodel "${_IMPORT_PREFIX}/lib/libGPUmodel.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
