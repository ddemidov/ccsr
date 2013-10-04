set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
include_directories(${CMAKE_SOURCE_DIR})

find_package(Boost COMPONENTS system)

add_executable(perf_benchmark perf_benchmark.cpp)
target_link_libraries(perf_benchmark ${Boost_LIBRARIES})
