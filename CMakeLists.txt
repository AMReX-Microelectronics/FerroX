# Parse FerroX version information
file(READ "${CMAKE_CURRENT_LIST_DIR}/dependencies.json" dependencies_data)
string(JSON ferrox_version GET "${dependencies_data}" version_ferrox)

# Preamble ####################################################################
#
cmake_minimum_required(VERSION 3.24.0)
project(FerroX VERSION ${ferrox_version})

include(${FerroX_SOURCE_DIR}/cmake/FerroXFunctions.cmake)

# In-source tree builds are messy and can screw up the build system.
# Avoid building at least in the same dir as the root dir:
if(CMAKE_BINARY_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "Building in-source is not supported! "
            "Create a build directory and remove "
            "${CMAKE_SOURCE_DIR}/CMakeCache.txt ${CMAKE_SOURCE_DIR}/CMakeFiles/")
endif()


# CMake policies ##############################################################
#
# Setting a cmake_policy to OLD is deprecated by definition and will raise a
# verbose warning
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)
endif()

# AMReX 21.06+ supports CUDA_ARCHITECTURES with CMake 3.20+
# CMake 3.18+: CMAKE_CUDA_ARCHITECTURES
# https://cmake.org/cmake/help/latest/policy/CMP0104.html
if(POLICY CMP0104)
    cmake_policy(SET CMP0104 OLD)
endif()


# C++ Standard in Superbuilds #################################################
#
# This is the easiest way to push up a C++17 requirement for AMReX, PICSAR and
# openPMD-api until they increase their requirement.
set_cxx17_superbuild()


# CCache Support ##############################################################
#
# this is an optional tool that stores compiled object files; allows fast
# re-builds even with "make clean" in between. Mainly used to store AMReX
# objects
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(FerroX_CCACHE_DEFAULT ON)
else()
    set(FerroX_CCACHE_DEFAULT OFF)  # we are a subproject in a superbuild
endif()
option(FerroX_CCACHE "Enable ccache for faster rebuilds" ${FerroX_CCACHE_DEFAULT})
if(FerroX_CCACHE)
    set_ccache()
endif()


# Output Directories ##########################################################
#
# temporary build directories
set_default_build_dirs()

# install directories
set_default_install_dirs()


# Options and Variants ########################################################
#
include(CMakeDependentOption)
option(FerroX_APP           "Build the FerroX executable application"     ON)
# Time-dependent support - matches Make.FerroX TIME_DEPENDENT=TRUE conditional compilation
# Default OFF matches GNU Make default behavior (only enabled when explicitly set)
option(FerroX_TIME_DEPENDENT "Time-dependent support" OFF)

# SUNDIALS integration support - matches Make.FerroX USE_SUNDIALS=TRUE conditional compilation
# Default OFF matches GNU Make default behavior (only enabled when explicitly set)
option(FerroX_SUNDIALS "SUNDIALS support" OFF)

# Conditional dependency logic matching Make.FerroX behavior
# SUNDIALS
#   builds SUNDIALS from source (default) or finds an existing install
#   only included when FerroX_SUNDIALS is enabled
if(FerroX_SUNDIALS)
    # Build/find SUNDIALS first, before AMReX looks for it
    include(${FerroX_SOURCE_DIR}/cmake/dependencies/SUNDIALS.cmake)

    # Enable AMReX SUNDIALS support when FerroX SUNDIALS is requested
    set(AMReX_SUNDIALS ON CACHE BOOL "Enable SUNDIALS interfaces" FORCE)
endif()

option(FerroX_EB            "Embedded boundary support"                  OFF)
option(FerroX_LIB           "Build FerroX as a library"                   OFF)
option(FerroX_MPI           "Multi-node support (message-passing)"       ON)
option(FerroX_SIMD          "CPU SIMD Acceleration"                      OFF)
option(FerroX_GPU_RDC       "Enable relocatable device code"             ON)

# Print debug options - matches Make.FerroX PRINT_* options
option(FerroX_PRINT_HIGH    "Enable high level debug printing"              OFF)
option(FerroX_PRINT_MEDIUM  "Enable medium level debug printing"            OFF)
option(FerroX_PRINT_LOW     "Enable low level debug printing"               OFF)
option(FerroX_PRINT_NAME    "Enable function name debug printing"           OFF)

# Warning options
option(FerroX_ENABLE_ALL_WARNINGS "Enable all compiler warnings"            OFF)

# Advanced option to run tests

# Advanced option to compile with the -g1 option for minimal debug symbols
# (useful to see, e.g., line numbers in backtraces)

set(FerroX_DIMS_VALUES 3)
set(FerroX_DIMS 3 CACHE STRING "Simulation dimensionality (3D only)")
list(REMOVE_DUPLICATES FerroX_DIMS)
foreach(D IN LISTS FerroX_DIMS)
    if(NOT D IN_LIST FerroX_DIMS_VALUES)
       message(FATAL_ERROR "FerroX_DIMS=${D} is not allowed."
          " FerroX only supports 3D simulations")
    endif()
endforeach()

# for some targets need to be triggered once, so any dim dependency will do
list(LENGTH FerroX_DIMS list_len)
math(EXPR list_last "${list_len} - 1")
list(GET FerroX_DIMS ${list_last} FerroX_DIMS_LAST)
ferrox_set_suffix_dims(FerroX_DIMS_LAST ${FerroX_DIMS_LAST})

set(FerroX_PRECISION_VALUES SINGLE DOUBLE)
set(FerroX_PRECISION DOUBLE CACHE STRING "Floating point precision (SINGLE/DOUBLE)")
set_property(CACHE FerroX_PRECISION PROPERTY STRINGS ${FerroX_PRECISION_VALUES})
if(NOT FerroX_PRECISION IN_LIST FerroX_PRECISION_VALUES)
    message(FATAL_ERROR "FerroX_PRECISION (${FerroX_PRECISION}) must be one of ${FerroX_PRECISION_VALUES}")
endif()

set(FerroX_PARTICLE_PRECISION_VALUES SINGLE DOUBLE)
set(FerroX_PARTICLE_PRECISION ${FerroX_PRECISION} CACHE STRING "Particle floating point precision (SINGLE/DOUBLE)")
set_property(CACHE FerroX_PARTICLE_PRECISION PROPERTY STRINGS ${FerroX_PARTICLE_PRECISION_VALUES})
if(NOT FerroX_PARTICLE_PRECISION IN_LIST FerroX_PARTICLE_PRECISION_VALUES)
    message(FATAL_ERROR "FerroX_PARTICLE_PRECISION (${FerroX_PARTICLE_PRECISION}) must be one of ${FerroX_PARTICLE_PRECISION_VALUES}")
endif()


set(FerroX_COMPUTE_VALUES NOACC OMP CUDA SYCL HIP)
set(FerroX_COMPUTE OMP CACHE STRING "On-node, accelerated computing backend (NOACC/OMP/CUDA/SYCL/HIP)")
set_property(CACHE FerroX_COMPUTE PROPERTY STRINGS ${FerroX_COMPUTE_VALUES})
if(NOT FerroX_COMPUTE IN_LIST FerroX_COMPUTE_VALUES)
    message(FATAL_ERROR "FerroX_COMPUTE (${FerroX_COMPUTE}) must be one of ${FerroX_COMPUTE_VALUES}")
endif()

option(FerroX_MPI_THREAD_MULTIPLE "MPI thread-multiple support, i.e. for async_io" ON)
mark_as_advanced(FerroX_MPI_THREAD_MULTIPLE)

option(FerroX_amrex_internal                    "Download & build AMReX" ON)

# PR testing options (primarily for CI/testing)
set(FerroX_amrex_pr "" CACHE STRING "AMReX pull request number for testing")
set(FerroX_ferrox_pr "" CACHE STRING "FerroX pull request number for testing")
mark_as_advanced(FerroX_amrex_pr)
mark_as_advanced(FerroX_ferrox_pr)

# change the default build type to Release (or RelWithDebInfo) instead of Debug
set_default_build_type("Release")

# Option to enable interprocedural optimization
# (also know as "link-time optimization" or "whole program optimization")
set(_FerroX_IPO_DEFAULT OFF)
if(DEFINED CMAKE_INTERPROCEDURAL_OPTIMIZATION)
    set(_FerroX_IPO_DEFAULT ${CMAKE_INTERPROCEDURAL_OPTIMIZATION})
endif()
option(FerroX_IPO
    "Compile FerroX with interprocedural optimization (will take more time)"
    ${_FerroX_IPO_DEFAULT}
)

# Unity builds combine all .cpp files into a single one (a single translation
# unit, TU).
option(FerroX_UNITY_BUILD          "FerroX library as unity build" OFF)


# enforce consistency of dependent options
if(FerroX_APP)
    set(FerroX_LIB ON CACHE STRING "Build FerroX as a library" FORCE)
endif()


# note: we could skip this if we solely build FerroX_APP, but if we build a
# shared FerroX library, then we need relocatable code.
option(ABLASTR_POSITION_INDEPENDENT_CODE
       "Build ABLASTR with position independent code" ON)
mark_as_advanced(ABLASTR_POSITION_INDEPENDENT_CODE)

# Define the variable BUILD_TESTING (ON by default),
# include CDash dashboard testing module
include(CTest)


# Dependencies ################################################################
#

# Old GCCs: std::filesystem as a separate library
try_compile(CXX_HAS_STD_FS "${FerroX_BINARY_DIR}/temp"
    "${FerroX_SOURCE_DIR}/cmake/has_filesystem.cpp"
    CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
)
try_compile(CXX_HAS_STD_FS_WITH_LINK "${FerroX_BINARY_DIR}/temp"
    "${FerroX_SOURCE_DIR}/cmake/has_filesystem.cpp"
    CMAKE_FLAGS -DCMAKE_CXX_STANDARD=17 -DCMAKE_CXX_STANDARD_REQUIRED=ON
    LINK_LIBRARIES stdc++fs
)

# AMReX
#   builds AMReX from source (default) or finds an existing install
set(FerroX_amrex_dim ${FerroX_DIMS})  # RZ is AMReX 2D
list(TRANSFORM FerroX_amrex_dim REPLACE RZ 2)
list(TRANSFORM FerroX_amrex_dim REPLACE RCYLINDER 1)
list(TRANSFORM FerroX_amrex_dim REPLACE RSPHERE 1)
list(REMOVE_DUPLICATES FerroX_amrex_dim)

include(${FerroX_SOURCE_DIR}/cmake/dependencies/AMReX.cmake)

foreach(D IN LISTS FerroX_amrex_dim)
    #   suppress warnings in AMReX headers (use -isystem instead of -I)
    ferrox_make_third_party_includes_system(AMReX::amrex_${D}d amrex_${D}d)
endforeach()
# For <AMReX_buildInfo.H>
include(AMReXBuildInfo)

# Targets #####################################################################
#
include(GenerateExportHeader)
set(_ALL_TARGETS)
foreach(D IN LISTS FerroX_DIMS)
    ferrox_set_suffix_dims(SD ${D})

    # ABLASTR library
#    add_library(ablastr_${SD})
#    set(_BUILDINFO_SRC ablastr_${SD})
#    list(APPEND _ALL_TARGETS ablastr_${SD})
#    add_library(FerroX::ablastr_${SD} ALIAS ablastr_${SD})

    # link into a library (default: static)
    if(FerroX_LIB)
        add_library(lib_${SD})
        add_library(FerroX::lib_${SD} ALIAS lib_${SD})
#        target_link_libraries(lib_${SD} PUBLIC ablastr_${SD})
        set(_BUILDINFO_SRC lib_${SD})
        list(APPEND _ALL_TARGETS lib_${SD})

        set_target_properties(lib_${SD} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            WINDOWS_EXPORT_ALL_SYMBOLS ON
        )

        # Optional: build only a single TU
        if(FerroX_UNITY_BUILD)
            set_target_properties(lib_${SD} PROPERTIES
                UNITY_BUILD ON
                UNITY_BUILD_MODE BATCH
                UNITY_BUILD_UNIQUE_ID "FERROX_UNITY_ID"
                # Number must be more than the number of .cpp files in FerroX
                UNITY_BUILD_BATCH_SIZE 10000
            )
        endif()

        # GCC < 9 need to manually link std::filesystem
#        if(NOT CXX_HAS_STD_FS)
#            if(CXX_HAS_STD_FS_WITH_LINK)
#                target_link_libraries(ablastr_${SD} PUBLIC stdc++fs)
#            else()
#                message(FATAL_ERROR "Your compiler does not support C++17 "
#                        "<filesystem> Please use a newer C++ compiler.")
#            endif()
#        endif()
    endif()

    # executable application
    #   note: we currently avoid a dependency on a core library
    #         for simpler usage, but could make this an option
    if(FerroX_APP)
        add_executable(app_${SD})
        add_executable(FerroX::app_${SD} ALIAS app_${SD})
        target_link_libraries(app_${SD} PRIVATE lib_${SD})
        set(_BUILDINFO_SRC app_${SD})
        list(APPEND _ALL_TARGETS app_${SD})
    endif()


    # own headers
#    target_include_directories(ablastr_${SD} PUBLIC
#        # future: own directory root
#        $<BUILD_INTERFACE:${FerroX_SOURCE_DIR}/Source>
#        $<BUILD_INTERFACE:${FerroX_BINARY_DIR}/Source>
#    )
    if(FerroX_LIB)
        target_include_directories(lib_${SD} PUBLIC
            $<BUILD_INTERFACE:${FerroX_SOURCE_DIR}/Source>
            $<BUILD_INTERFACE:${FerroX_BINARY_DIR}/Source>
            $<BUILD_INTERFACE:${FerroX_SOURCE_DIR}/Source/Utils>
            $<BUILD_INTERFACE:${FerroX_SOURCE_DIR}/Source/Input>
            $<BUILD_INTERFACE:${FerroX_SOURCE_DIR}/Source/Input/GeometryProperties>
            $<BUILD_INTERFACE:${FerroX_SOURCE_DIR}/Source/Input/BoundaryConditions>
            $<BUILD_INTERFACE:${FerroX_SOURCE_DIR}/Source/Solver>
            $<BUILD_INTERFACE:${FerroX_SOURCE_DIR}/Source/Utils/SelectWarpXUtils>
            $<BUILD_INTERFACE:${FerroX_SOURCE_DIR}/Source/Utils/FerroXUtils>
            $<BUILD_INTERFACE:${FerroX_SOURCE_DIR}/Source/Utils/eXstaticUtils>
        )
    endif()


    if(FerroX_LIB)
        # if we include <AMReX_buildInfo.H> we will need to call:
        generate_buildinfo(${_BUILDINFO_SRC} "${FerroX_SOURCE_DIR}")
        target_link_libraries(lib_${SD} PRIVATE buildInfo::${_BUILDINFO_SRC})
        unset(_BUILDINFO_SRC)

        # add sources
        target_sources(lib_${SD} PRIVATE Source/FerroX.cpp)
    endif()

    # add sources
    if(FerroX_APP)
        target_sources(app_${SD} PRIVATE Source/main.cpp)
    endif()
endforeach()

# Headers controlling symbol visibility (for Windows)
# Note: once FerroX gets rid of all its globals, we will not need this.
#   - extern ... functions
#   - static class member variables
#generate_export_header(ablastr_${FerroX_DIMS_LAST}
#    BASE_NAME ablastr
#    EXPORT_FILE_NAME Source/ablastr/export.H)
if(FerroX_LIB)
    generate_export_header(lib_${FerroX_DIMS_LAST}
        BASE_NAME ferrox
        EXPORT_FILE_NAME Source/Utils/export.H)
endif()
# At build-time, we might need to set FerroX_<last>_EXPORTS and ablastr_<last>_EXPORTS on all targets
foreach(D IN LISTS FerroX_DIMS)
    ferrox_set_suffix_dims(SD ${D})
#    target_compile_definitions(ablastr_${SD} PRIVATE ablastr_${FerroX_DIMS_LAST}_EXPORTS)
    if(FerroX_LIB)
        target_compile_definitions(lib_${SD} PRIVATE lib_${FerroX_DIMS_LAST}_EXPORTS)  # note: some collision risk here
    endif()
    # Static libs: FERROX_STATIC_DEFINE and ABLASTR_STATIC_DEFINE during build time
    # note: using a static lib (lib_${SD}) in a .dll (pyFerroX_${SD}) still needs the exports for
    #       dllimport at DLL build time.
    #       https://cmake.org/cmake/help/latest/prop_tgt/WINDOWS_EXPORT_ALL_SYMBOLS.html
    if(NOT BUILD_SHARED_LIBS)
#        target_compile_definitions(ablastr_${SD} PRIVATE ABLASTR_STATIC_DEFINE)
        if(FerroX_LIB)
            target_compile_definitions(lib_${SD} PRIVATE FERROX_STATIC_DEFINE)
        endif()
    endif()
endforeach()

# Add after finding AMReX and SUNDIALS packages
if(FerroX_SUNDIALS AND AMReX_FOUND)
    message(STATUS "Checking AMReX/SUNDIALS type compatibility...")
    message(STATUS "DEBUG: FerroX_PRECISION=${FerroX_PRECISION}, SUNDIALS_PRECISION=${SUNDIALS_PRECISION}")

    file(WRITE ${CMAKE_BINARY_DIR}/test_types.cpp "#include <AMReX_REAL.H>\n#include <sundials/sundials_types.h>\n#include <type_traits>\n#include <iostream>\nint main() {\n    static_assert(sizeof(amrex::Real) == sizeof(sunrealtype));\n    static_assert(std::is_same<amrex::Real, sunrealtype>::value);\n    static_assert(sizeof(int) == sizeof(sunindextype));\n    std::cout << sizeof(amrex::Real) << 32 << sizeof(sunrealtype) << 10;\n    std::cout << sizeof(int) << 32 << sizeof(sunindextype) << 10;\n    return 0;\n}")

    try_compile(
        TYPES_COMPATIBLE
        ${CMAKE_BINARY_DIR}/test_build
        ${CMAKE_BINARY_DIR}/test_types.cpp
        CMAKE_FLAGS
            "-DINCLUDE_DIRECTORIES=${AMReX_INCLUDE_DIRS};${SUNDIALS_INCLUDE_DIR}"
        OUTPUT_VARIABLE COMPILE_OUTPUT
    )

endif()


if(FerroX_LIB)
    add_subdirectory(Source)
    add_subdirectory(Source/Input)
    add_subdirectory(Source/Solver)
    add_subdirectory(Source/Utils)
endif()

# Interprocedural optimization (IPO) / Link-Time Optimization (LTO)
if(FerroX_IPO)
    ferrox_enable_IPO("${_ALL_TARGETS}")
endif()

# link dependencies
foreach(D IN LISTS FerroX_DIMS)
    ferrox_set_suffix_dims(SD ${D})

    # Link AMReX to lib target since we don't have ablastr targets
    if(FerroX_LIB)
        if(D STREQUAL "RZ")
            target_link_libraries(lib_${SD} PUBLIC FerroX::thirdparty::amrex_2d)
        elseif(D STREQUAL "RCYLINDER")
            target_link_libraries(lib_${SD} PUBLIC FerroX::thirdparty::amrex_1d)
        elseif(D STREQUAL "RSPHERE")
            target_link_libraries(lib_${SD} PUBLIC FerroX::thirdparty::amrex_1d)
        else()
            target_link_libraries(lib_${SD} PUBLIC FerroX::thirdparty::amrex_${D}d)
        endif()
    endif()




endforeach()

# C++ properties: at least a C++17 capable compiler is needed
if(FerroX_COMPUTE STREQUAL CUDA)
    # AMReX helper function: propagate CUDA specific target & source properties
    foreach(ferrox_tgt IN LISTS _ALL_TARGETS)
        setup_target_for_cuda_compilation(${ferrox_tgt})
    endforeach()
    foreach(ferrox_tgt IN LISTS _ALL_TARGETS)
        target_compile_features(${ferrox_tgt} PUBLIC cuda_std_17)
    endforeach()
    set_target_properties(${_ALL_TARGETS} PROPERTIES
        CUDA_EXTENSIONS OFF
        CUDA_STANDARD_REQUIRED ON
    )
else()
    foreach(ferrox_tgt IN LISTS _ALL_TARGETS)
        target_compile_features(${ferrox_tgt} PUBLIC cxx_std_17)
    endforeach()
    set_target_properties(${_ALL_TARGETS} PROPERTIES
        CXX_EXTENSIONS OFF
        CXX_STANDARD_REQUIRED ON
    )
endif()

foreach(D IN LISTS FerroX_DIMS)
    ferrox_set_suffix_dims(SD ${D})

    # fancy binary name for build variants
    ferrox_set_binary_name(${D})
endforeach()


# Defines #####################################################################
#
foreach(D IN LISTS FerroX_DIMS)
    ferrox_set_suffix_dims(SD ${D})

    # Time-dependent compilation support - matches Make.FerroX TIME_DEPENDENT=TRUE
    if(FerroX_TIME_DEPENDENT)
        target_compile_definitions(lib_${SD} PUBLIC TIME_DEPENDENT)
    endif()

    # SUNDIALS compilation support - matches Make.FerroX USE_SUNDIALS=TRUE
    if(FerroX_SUNDIALS)
        target_compile_definitions(lib_${SD} PUBLIC USE_SUNDIALS)

        # Link SUNDIALS 7+ core component if available (following AMReX pattern)
        if(SUNDIALS_VERSION VERSION_GREATER_EQUAL 7)
            target_link_libraries(lib_${SD} PUBLIC SUNDIALS::core)
        endif()

        target_link_libraries(lib_${SD} PUBLIC
            SUNDIALS::cvode
            SUNDIALS::arkode
            SUNDIALS::nvecserial
            SUNDIALS::nvecmanyvector
        )

        if(FerroX_MPI)
            target_link_libraries(lib_${SD} PUBLIC
                SUNDIALS::nvecmpimanyvector
            )
        endif()

        # Add backend-specific libraries
        if(FerroX_COMPUTE STREQUAL OMP)
            if(TARGET SUNDIALS::nvecopenmp)
                target_link_libraries(lib_${SD} PUBLIC SUNDIALS::nvecopenmp)
            endif()
        elseif(FerroX_COMPUTE STREQUAL CUDA)
            if(TARGET SUNDIALS::nveccuda)
                target_link_libraries(lib_${SD} PUBLIC SUNDIALS::nveccuda)
            endif()
            # Conditionally link fused CUDA libraries if available
            if(TARGET SUNDIALS::cvode_fused_cuda)
                target_link_libraries(lib_${SD} PUBLIC SUNDIALS::cvode_fused_cuda)
            endif()
            # Also add ARKODE fused kernels if using ARKODE with CUDA
            if(TARGET SUNDIALS::arkode_fused_cuda)
                target_link_libraries(lib_${SD} PUBLIC SUNDIALS::arkode_fused_cuda)
            endif()
        elseif(FerroX_COMPUTE STREQUAL HIP)
            if(TARGET SUNDIALS::nvechip)
                target_link_libraries(lib_${SD} PUBLIC SUNDIALS::nvechip)
            endif()
            # Conditionally link fused HIP libraries if available
            if(TARGET SUNDIALS::cvode_fused_hip)
                target_link_libraries(lib_${SD} PUBLIC SUNDIALS::cvode_fused_hip)
            endif()
            # Also add ARKODE fused kernels if using ARKODE with HIP
            if(TARGET SUNDIALS::arkode_fused_hip)
                target_link_libraries(lib_${SD} PUBLIC SUNDIALS::arkode_fused_hip)
            endif()
        elseif(FerroX_COMPUTE STREQUAL SYCL)
            if(TARGET SUNDIALS::nvecsycl)
                target_link_libraries(lib_${SD} PUBLIC SUNDIALS::nvecsycl)
            endif()
        endif()
    endif()

    # Print debug options - matches Make.FerroX PRINT_* options
    if(FerroX_PRINT_HIGH)
        target_compile_definitions(lib_${SD} PUBLIC PRINT_HIGH PRINT_MEDIUM PRINT_LOW PRINT_NAME)
    endif()
    if(FerroX_PRINT_MEDIUM)
        target_compile_definitions(lib_${SD} PUBLIC PRINT_MEDIUM PRINT_LOW PRINT_NAME)
    endif()
    if(FerroX_PRINT_LOW)
        target_compile_definitions(lib_${SD} PUBLIC PRINT_LOW PRINT_NAME)
    endif()
    if(FerroX_PRINT_NAME)
        target_compile_definitions(lib_${SD} PUBLIC PRINT_NAME)
    endif()


    # <cmath>: M_PI
#    if(WIN32)
#        target_compile_definitions(ablastr_${SD} PUBLIC _USE_MATH_DEFINES)
#    endif()

    # Windows DLLs and Global Symbols
    # https://stackoverflow.com/questions/54560832/cmake-windows-export-all-symbols-does-not-cover-global-variables/54568678#54568678
    #
#    if(WIN32 AND BUILD_SHARED_LIBS)
#        set(ABLASTR_IS_DLL ON)
#        target_compile_definitions(ablastr_${SD} PRIVATE ABLASTR_IS_DLL_BUILDING)
#    endif()
#    if(WIN32 AND FerroX_LIB AND BUILD_SHARED_LIBS)
#        set(ABLASTR_IS_DLL ON)
#        target_compile_definitions(lib_${SD} PRIVATE ABLASTR_IS_DLL_BUILDING)
#    endif()
#    if(WIN32 AND FerroX_PYTHON)
#        set(ABLASTR_IS_DLL ON)
#        target_compile_definitions(pyFerroX_${SD} PRIVATE ABLASTR_IS_DLL_BUILDING)
#    endif()
endforeach()


# Warnings ####################################################################
#
set_cxx_warnings()
set_cxx_warning_flags()


# Generate Configuration and .pc Files ########################################
#
get_source_version(FerroX_${FerroX_DIMS_LAST} ${FerroX_SOURCE_DIR})
set(FerroX_GIT_VERSION ${FerroX_${FerroX_DIMS_LAST}_GIT_VERSION})
configure_file(
    ${FerroX_SOURCE_DIR}/Source/Utils/FerroXVersion.H.in
    ${FerroX_BINARY_DIR}/Source/Utils/FerroXVersion.H
    @ONLY
)

# these files are used if FerroX is installed and picked up by a downstream
# project (not needed yet)

#include(CMakePackageConfigHelpers)
#write_basic_package_version_file("FerroXConfigVersion.cmake"
#    VERSION ${FerroX_VERSION}
#    COMPATIBILITY SameMajorVersion
#)


# Installs ####################################################################
#
# headers, libraries and executables
set(FerroX_INSTALL_TARGET_NAMES)
foreach(D IN LISTS FerroX_DIMS)
    ferrox_set_suffix_dims(SD ${D})
#    list(APPEND FerroX_INSTALL_TARGET_NAMES ablastr_${SD})
    if(FerroX_APP)
        list(APPEND FerroX_INSTALL_TARGET_NAMES app_${SD})
    endif()
    if(FerroX_LIB)
        list(APPEND FerroX_INSTALL_TARGET_NAMES lib_${SD})
    endif()

    install(TARGETS ${FerroX_INSTALL_TARGET_NAMES}
        EXPORT FerroXTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )

    # simplified library alias
    # this is currently expected by Python bindings
    if(FerroX_LIB)
        if(IS_ABSOLUTE ${CMAKE_INSTALL_LIBDIR})
            set(ABS_INSTALL_LIB_DIR "${CMAKE_INSTALL_LIBDIR}")
        else()
            set(ABS_INSTALL_LIB_DIR "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
        endif()
        # escape spaces for generated cmake_install.cmake file
        file(TO_CMAKE_PATH "${ABS_INSTALL_LIB_DIR}" ABS_INSTALL_LIB_DIR)

        install(CODE "file(CREATE_LINK
            $<TARGET_FILE_NAME:lib_${SD}>
            \"${ABS_INSTALL_LIB_DIR}/libferrox.${SD}$<TARGET_FILE_SUFFIX:lib_${SD}>\"
            COPY_ON_ERROR SYMBOLIC)")
    endif()

    # FerroX
    # Utils/FerroXVersion.H

    # ABLASTR
    # export.H

    # CMake package file for find_package(FerroX::FerroX) in depending projects
    #install(EXPORT FerroXTargets
    #    FILE FerroXTargets.cmake
    #    NAMESPACE FerroX::
    #    DESTINATION ${FerroX_INSTALL_CMAKEDIR}
    #)
    #install(
    #    FILES
    #        ${FerroX_BINARY_DIR}/FerroXConfig.cmake
    #        ${FerroX_BINARY_DIR}/FerroXConfigVersion.cmake
    #    DESTINATION ${FerroX_INSTALL_CMAKEDIR}
    #)
endforeach()




# Tests #######################################################################
#
#if(BUILD_TESTING)
#    enable_testing()
#    if(FerroX_APP OR FerroX_PYTHON)
#        add_subdirectory(Exec/Examples)
#    endif()
#endif()


# Status Summary for Build Options ############################################
#
ferrox_print_summary()
