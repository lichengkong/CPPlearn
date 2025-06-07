# cmake_minimum_required(VERSION 3.18)
project(plugin_app)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
add_definitions(-w)

set(THREADS_PREFER_PTHREAD_FLAG ON)

# find_package(Threads REQUIRED)
include_directories(
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparties"
        "${CMAKE_CURRENT_SOURCE_DIR}/thirdparties/rest_rpc"  # 第三方依赖，直接引入源码

        # "${CMAKE_CURRENT_SOURCE_DIR}/thirdparties/asio"  #sudo apt -y install libasio-dev
        # "${CMAKE_CURRENT_SOURCE_DIR}/thirdparties/boost" #sudo apt -y install libboost-all-dev
        # "${CMAKE_CURRENT_SOURCE_DIR}/thirdparties/msgpack"  #sudo apt -y install libmsgpack-dev
)

add_library(custom SHARED plugin/custom_plugin.cc)
add_library(dummy SHARED plugin/dummy_plugin.cc)

add_executable(plugin_app plugin_app/main.cpp)
add_executable(plugin_app_test plugin_app/test.cpp)

target_link_libraries(plugin_app boost_system boost_filesystem pthread dl)
target_link_libraries(plugin_app_test boost_system boost_filesystem pthread dl)
add_executable(client client/app_client.cpp)
target_link_libraries(client  pthread)