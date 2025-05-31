# cmake_minimum_required(VERSION 3.10.0)
project(threadlogger VERSION 0.1.0 LANGUAGES C CXX)
add_compile_options(-fcoroutines)

add_executable(threadlogger main.cpp)

target_link_libraries(threadlogger pthread)

