add_executable(asyncAPI asyncAPI/asyncAPI.cu)

add_executable(cxx11_cuda c++11_cuda/c++11_cuda.cu)
# set_target_properties(cxx11_cuda PROPERTIES 
    # CXX_STANDARD 11
    # CXX_STANDARD_REQUIRED ON
    # CXX_EXTENSIONS OFF
# )
target_compile_options(cxx11_cuda PRIVATE "-std=c++11") # TODO fix warning
# file(COPY_FILE c++11_cuda/warandpeace.txt build/c++11_cuda/warandpeace.txt)

add_executable(clock clock/clock.cu)
# add_executable(clock_nvrtc clock_nvrtc/clock_kernel.cu clock_nvrtc/clock.cpp)
# target_link_libraries(clock_nvrtc PUBLIC CUDA::nvrtc)

# add_executable(cudaOpenMP cudaOpenMP/cudaOpenMP.cu)
# find_package(OpenMP REQUIRED)
# if(OpenMP_CXX_FOUND)
#     message(STATUS "Found OpenMP CXX runtime")
# endif()
# # message(status ${OpenMP_DIR})
# target_link_libraries(cudaOpenMP PRIVATE OpenMP::OpenMP_CXX)

add_executable(fp16ScalarProduct fp16ScalarProduct/fp16ScalarProduct.cu)

# add_executable(matrixMulDynlinkJIT 
#     matrixMulDynlinkJIT/cuda_drvapi_dynlink.c  
#     matrixMulDynlinkJIT/matrixMulDynlinkJIT.cpp 
#     matrixMulDynlinkJIT/matrixMul_gold.cpp 
#     # matrixMulDynlinkJIT/matrixMul_kernel_32_ptxdump.c 
#     matrixMulDynlinkJIT/matrixMul_kernel_64_ptxdump.c
# )
# # set_target_properties(matrixMulDynlinkJIT PROPERTIES
# #     POSITION_INDEPENDENT_CODE ON
# # )
# set_target_properties(matrixMulDynlinkJIT PROPERTIES LANGUAGE CXX)
# target_compile_options(matrixMulDynlinkJIT PRIVATE "-Xcompiler -fPIE")
# # set_target_properties(matrixMulDynlinkJIT PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# target_link_libraries(matrixMulDynlinkJIT PRIVATE
#     CUDA::nvrtc         # nvrtc 支持
#     cuda                # cuDeviceGet 等驱动 API
# )


