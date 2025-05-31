project(coroproj VERSION 0.1.0 LANGUAGES C CXX)

add_executable(coroproj corodemo.cpp)
target_link_libraries(coroproj  pthread)