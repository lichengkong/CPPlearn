cmake_minimum_required(VERSION 3.18.0)
project(timer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_INSTALL_PREFIX install)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)



# threadlogger
add_subdirectory(threadlogger)

# plugincpp
add_subdirectory(plugin)

# timer
add_subdirectory(timer)

# coroutine raw C++ 20
add_subdirectory(coroproj)