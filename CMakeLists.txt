cmake_minimum_required(VERSION 3.10)
project(ex3___c C)

set(CMAKE_C_STANDARD 11)

add_executable(ex3___c main.c)

# Requires libibverbs-dev (or equivalent) on the target Linux/RDMA host,
# e.g. the mlxstudNN cluster. Not buildable on Windows -- there is no
# Verbs implementation there.
target_link_libraries(ex3___c ibverbs m)
