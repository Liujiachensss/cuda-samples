message(STATUS ${CUDA_LIBRARIES})
add_executable(deviceQ deviceQuery/deviceQuery.cpp)
target_link_libraries(deviceQ PUBLIC cudart)
target_compile_options(deviceQ PRIVATE --ccbin)