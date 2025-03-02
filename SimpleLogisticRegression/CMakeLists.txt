cmake_minimum_required(VERSION 3.10)
project(SimpleLogisticRegression)

set(CMAKE_CXX_STANDARD 14)

#eigen
find_package (EIGEN3 REQUIRED)
if(EIGEN3_FOUND)
  message(STATUS "Found EIGEN3 version: ${EIGEN3_VERSION}")
  message(STATUS "EIGEN3 include directories: ${EIGEN3_INCLUDE_DIRS}")
  include_directories(${EIGEN3_INCLUDE_DIRS})
  SET( EIGEN3_INCLUDE_DIR "$ENV{EIGEN3_INCLUDE_DIR}" )
else()
  message(FATAL_ERROR "EIGEN3 not found, please read the README.md")
endif(EIGEN3_FOUND)

add_executable(SimpleLogisticRegression SimpleLogisticRegression.cpp)
target_link_libraries(SimpleLogisticRegression Eigen3::Eigen)