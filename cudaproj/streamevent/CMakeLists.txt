project(streamevent VERSION 0.1.0 LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_EXPERIMENTAL_CXX_MODULES ON)  
# set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)   
# add_definitions("-g -G")
set(CMAKE_BUILD_TYPE Debug)

add_executable(${PROJECT_NAME} cudastream.cu)

target_include_directories(${PROJECT_NAME} PUBLIC
    /usr/local/cuda/include
    /usr/local/TensorRT/include)

target_link_directories(${PROJECT_NAME} PUBLIC
    /usr/local/cuda/lib64
    /usr/local/TensorRT/lib)

target_link_libraries(${PROJECT_NAME} PUBLIC nvinfer cudart dl pthread)
