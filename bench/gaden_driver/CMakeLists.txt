cmake_minimum_required(VERSION 3.16)
project(gaden_driver CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

set(GADEN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../external/gaden_core)
add_subdirectory(${GADEN_ROOT} gaden_core EXCLUDE_FROM_ALL)

add_executable(gaden_driver main.cpp)
target_link_libraries(gaden_driver gaden)
