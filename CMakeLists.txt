cmake_minimum_required (VERSION 2.6)
project(ale-op)

execute_process(
  COMMAND python -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
  OUTPUT_VARIABLE TF_INCLUDE
  RESULT_VARIABLE RV
  ERROR_VARIABLE TF_ERROR
  OUTPUT_VARIABLE TF_INCLUDE
  )
message(STATUS TF_INCLUDES:  ${TF_INCLUDE})
set(ALE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/../Arcade-Learning-Environment/src CACHE PATH "ALE headers")
set(ALE_LIBRARY_DIR ${CMAKE_SOURCE_DIR}/../Arcade-Learning-Environment CACHE PATH "ALE shared library")
set(ROM_PATH ${CMAKE_SOURCE_DIR}/roms CACHE PATH "Path to Atari rom files")
add_definitions(-DROM_PATH="${ROM_PATH}/")

include_directories(${TF_INCLUDE})
include_directories(${ALE_INCLUDE_DIR})
link_directories(${ALE_LIBRARY_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wunused -Wno-deprecated-declarations -fPIC -O3 -fomit-frame-pointer -D_GLIBCXX_USE_CXX11_ABI=0")

add_library(aleop SHARED src/ale.cc )
target_link_libraries(aleop ale)
