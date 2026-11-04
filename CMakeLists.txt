cmake_minimum_required(VERSION 3.10)
project(HCP-project C)

set(CMAKE_C_STANDARD 11)

add_executable(HCP-project main.c)

# Requires libibverbs-dev (or equivalent) on the target Linux/RDMA host,
# e.g. the mlxstudNN cluster. Not buildable on Windows -- there is no
# Verbs implementation there.
target_link_libraries(HCP-project ibverbs m)
