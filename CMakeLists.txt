cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED fftw3)


set(SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/random_error.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/power.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/transport.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/plot.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Src/main.cpp

)

set(HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/Inc/complex_types.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Inc/filter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Inc/power.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Inc/random_error.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Inc/transport.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Inc/plot.h
)

# include files
include_directories(
    ${FFTW_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/Inc
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${FFTW_LIBRARIES} ${OpenCV_LIBS})
