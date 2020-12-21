project("XOR")
CMAKE_MINIMUM_REQUIRED(VERSION 3.19)

message(STATUS "DYNET_BASE = $ENV{DYNET_BASE}")

set(CMAKE_CXX_FLAGS "-std=c++11")
set(RELEASE_OPT_LEVEL 3)

include_directories($ENV{DYNET_BASE})
add_executable(xor train_xor-batch.cc)
target_link_libraries(xor -L$ENV{DYNET_BASE}/build/dynet -ldynet)
