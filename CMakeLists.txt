cmake_minimum_required(VERSION 3.10)

project(NeuralCore)

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

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src)

# Valgrind memcheck support
find_program(VALGRIND "valgrind")
if(VALGRIND)
    message(STATUS "Valgrind found: ${VALGRIND}")
    set(MEMORYCHECK_COMMAND ${VALGRIND})
    set(MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full --show-leak-kinds=all --error-exitcode=1")
    include(CTest)
else()
    message(STATUS "Valgrind not found — memcheck target will not be available")
    enable_testing()
endif()

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(lib/googletest)

# Custom target for convenient 'make valgrind_check'
if(VALGRIND)
    add_custom_target(valgrind_check
        COMMAND ${CMAKE_CTEST_COMMAND} -T memcheck --output-on-failure
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        DEPENDS ${CMAKE_PROJECT_NAME}_tests
        COMMENT "Running tests under Valgrind..."
    )
endif()

# Add additional flags for debug builds
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wpedantic")
