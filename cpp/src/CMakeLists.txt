# cpp/src/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(Trademore VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


include(FetchContent)

FetchContent_Declare(
    cpr
    GIT_REPOSITORY https://github.com/libcpr/cpr.git
    GIT_TAG 1.10.5  #
)

FetchContent_MakeAvailable(cpr)


FetchContent_Declare(
    cli11_proj
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG v2.3.2
)

FetchContent_MakeAvailable(cli11_proj)

FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.2  
)

FetchContent_MakeAvailable(json)

add_executable(alpaca alpaca.cpp)

target_link_libraries(alpaca PRIVATE cpr::cpr CLI11::CLI11 nlohmann_json::nlohmann_json)
