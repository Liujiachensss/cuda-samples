add_executable(convolutionFFT2D 
               convolutionFFT2D/main.cpp
               convolutionFFT2D/convolutionFFT2D_gold.cpp
               convolutionFFT2D/convolutionFFT2D.cu
)
target_include_directories(convolutionFFT2D PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/convolutionFFT2D)
target_link_libraries(convolutionFFT2D PUBLIC CUDA::cufft)


add_executable(FDTD3d 
               FDTD3d/src/FDTD3d.cpp
               FDTD3d/src/FDTD3dReference.cpp
               FDTD3d/src/FDTD3dGPU.cu
)
target_include_directories(FDTD3d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/FDTD3d/inc)