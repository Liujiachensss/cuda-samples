add_executable(asyncAPI asyncAPI/asyncAPI.cu)

add_executable(cxx11_cuda c++11_cuda/c++11_cuda.cu)
# file(COPY_FILE c++11_cuda/warandpeace.txt build/c++11_cuda/warandpeace.txt)

add_executable(clock clock/clock.cu)
# add_executable(clock_nvrtc clock_nvrtc/clock_kernel.cu clock_nvrtc/clock.cpp)
# target_link_libraries(clock_nvrtc PUBLIC CUDA::nvrtc)

add_executable(cudaOpenMP cudaOpenMP/cudaOpenMP.cu)
find_package(OpenMP REQUIRED)
target_link_libraries(cudaOpenMP PRIVATE OpenMP::OpenMP_CXX)

