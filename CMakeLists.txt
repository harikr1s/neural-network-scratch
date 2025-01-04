cmake_minimum_required(VERSION 3.5.0 FATAL_ERROR)
project(neural-network-cpp VERSION 0.1.0 LANGUAGES CXX)
add_executable(main src/main.cpp)
set_property(TARGET main PROPERTY CXX_STANDARD 17)

include(FetchContent)

FetchContent_Declare(
    cxxplot
    GIT_REPOSITORY https://github.com/USNavalResearchLaboratory/cxxplot.git
    GIT_TAG v0.4.1
)
FetchContent_MakeAvailable(cxxplot)

target_link_libraries(main PRIVATE cxxplot)