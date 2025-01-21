add_executable(convolutionSeparable convolutionSeparable/main.cpp 
                                    convolutionSeparable/convolutionSeparable_gold.cpp 
                                    convolutionSeparable/convolutionSeparable.cu)
target_include_directories(convolutionSeparable PUBLIC convolutionSeparable)
target_link_libraries(convolutionSeparable PUBLIC CUDA::cudart)