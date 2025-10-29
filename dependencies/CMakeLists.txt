cmake_minimum_required(VERSION 3.21)
project(dependency-integration)

##############################################################
# Dependencies
##############################################################

# Configure clingo options before adding subdirectory
add_subdirectory(abseil)
add_subdirectory(fmt)
add_subdirectory(gtl)
add_subdirectory(oneTBB)
add_subdirectory(valla)


set(CLINGO_BUILD_WITH_PYTHON "OFF" CACHE STRING "Disable Python support")
set(CLINGO_BUILD_WITH_LUA "OFF" CACHE STRING "Disable Lua support")
set(CLINGO_BUILD_APPS "OFF" CACHE BOOL "Disable building clingo applications")
set(CLINGO_BUILD_TESTS "OFF" CACHE BOOL "Disable clingo tests")
set(CLINGO_BUILD_EXAMPLES "OFF" CACHE BOOL "Disable clingo examples")
add_subdirectory(clingo)