cmake_minimum_required(VERSION 3.10)

project(UwU_Learner)

# Enable sanitizers for debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Enabling sanitizers for Debug build")
    set(SANITIZER_FLAGS "-fsanitize=address,undefined -fno-omit-frame-pointer -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_FLAGS}")
endif()

# Enable gprof profiler for profiling builds
if(CMAKE_BUILD_TYPE STREQUAL "Profiling")
    message(STATUS "Enabling gprof profiler for Profiling build")
    set(PROFILER_FLAGS "-pg")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PROFILER_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${PROFILER_FLAGS}")
endif()

# Enable optimizations for release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Enabling optimizations for Release build")
    set(OPTIMIZATION_FLAGS "-O3 -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTIMIZATION_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OPTIMIZATION_FLAGS}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable OpenMP if available
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found. Enabling OpenMP support.")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
else()
    message(WARNING "OpenMP not found. The project will compile without parallelization.")
endif()

include_directories(src)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(lib/googletest)
#add_subdirectory(lib/Arcade-Learning-Environment-0.6.1)

# Add additional flags for debug builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wpedantic")
