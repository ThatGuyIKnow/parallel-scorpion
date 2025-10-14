cmake_minimum_required(VERSION 3.21)
project(dependency-integration)

##############################################################
# Dependencies
##############################################################

add_subdirectory(abseil)
add_subdirectory(fmt)
add_subdirectory(gtl)
add_subdirectory(oneTBB)
add_subdirectory(valla)
