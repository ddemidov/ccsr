set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
include_directories(${CMAKE_SOURCE_DIR})

find_package(Boost COMPONENTS system)
add_executable(hello hello.cpp)
target_link_libraries(hello ${Boost_LIBRARIES})
