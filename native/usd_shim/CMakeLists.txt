cmake_minimum_required(VERSION 3.19)
project(usd_shim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED USD_ROOT)
    set(USD_ROOT "/Users/varomix/dev/OpenUSD" CACHE PATH "Path to a built OpenUSD install (has include/pxr, lib/, pxrConfig.cmake)")
endif()

list(APPEND CMAKE_PREFIX_PATH "${USD_ROOT}")
find_package(pxr REQUIRED CONFIG PATHS "${USD_ROOT}")

add_library(usd_shim SHARED usd_shim.cpp)

target_include_directories(usd_shim PRIVATE ${PXR_INCLUDE_DIRS})
target_link_libraries(usd_shim PRIVATE
    usd
    usdGeom
    usdShade
    usdLux
    sdf
    ar
    tf
    vt
    gf
)

# OpenSubdiv (Far) for subdivision-surface refinement. It ships inside the
# same OpenUSD install (USD depends on it), so reuse those headers/lib rather
# than requiring a separate find_package.
target_include_directories(usd_shim PRIVATE "${USD_ROOT}/include")
find_library(OSD_CPU_LIB osdCPU PATHS "${USD_ROOT}/lib" REQUIRED)
target_link_libraries(usd_shim PRIVATE ${OSD_CPU_LIB})

# This USD build has Python support baked into the core libraries (Python
# symbols are referenced via flat-namespace dynamic lookup, not a normal
# LC_LOAD_DYLIB dependency), so a pure-C++ host process crashes on load
# with "symbol not found in flat namespace" unless something in the
# process provides them. Link libpython directly so it's loaded eagerly
# and gets vendored by build.sh like any other dependency, even though
# the shim itself never calls into Python.
find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Development)
target_link_libraries(usd_shim PRIVATE Python3::Python)

# Rpath is set so the shim resolves its USD/TBB dependencies from the
# vendored copies under lib/darwin/usd/ next to the built dylib, not from
# the local OpenUSD build dir this was compiled against. build.sh copies
# the dylibs into place and drops libusd_shim.dylib in lib/darwin/ after
# this target builds.
#
# The install name is @rpath-relative *including* the lib/darwin prefix:
# Odin links the lumbre executable with a single LC_RPATH of @loader_path
# (the directory holding the binary, i.e. the repo root), so the shim has
# to announce itself as lib/darwin/libusd_shim.dylib for dyld to find it.
# Keeps `odin build .` working with no extra linker flags.
set_target_properties(usd_shim PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "@loader_path/usd"
    INSTALL_NAME_DIR "@rpath/lib/darwin"
)
