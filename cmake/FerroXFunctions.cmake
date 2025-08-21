# CMake functions for FerroX build system
# Adapted from FerroX CMake functions
# Original source: https://github.com/ECP-FerroX/FerroX

# Set C++17 for the whole build if not otherwise requested
#
# This is the easiest way to push up a C++17 requirement for AMReX, PICSAR and
# openPMD-api until they increase their requirement.
#
macro(set_cxx17_superbuild)
    if(NOT DEFINED CMAKE_CXX_STANDARD)
        set(CMAKE_CXX_STANDARD 17)
    endif()
    if(NOT DEFINED CMAKE_CXX_EXTENSIONS)
        set(CMAKE_CXX_EXTENSIONS OFF)
    endif()
    if(NOT DEFINED CMAKE_CXX_STANDARD_REQUIRED)
        set(CMAKE_CXX_STANDARD_REQUIRED ON)
    endif()

    if(NOT DEFINED CMAKE_CUDA_STANDARD)
        set(CMAKE_CUDA_STANDARD 17)
    endif()
    if(NOT DEFINED CMAKE_CUDA_EXTENSIONS)
        set(CMAKE_CUDA_EXTENSIONS OFF)
    endif()
    if(NOT DEFINED CMAKE_CUDA_STANDARD_REQUIRED)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    endif()
endmacro()


# find the CCache tool and use it if found
#
macro(set_ccache)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        if(FerroX_COMPUTE STREQUAL CUDA)
            set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        endif()
        message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
    else()
        message(STATUS "Could NOT find CCache")
    endif()
    mark_as_advanced(CCACHE_PROGRAM)
endmacro()


# set names and paths of temporary build directories
# the defaults in CMake are sub-ideal for historic reasons, lets make them more
# Unix-ish and portable.
#
macro(set_default_build_dirs)
    if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                CACHE PATH "Build directory for archives")
        mark_as_advanced(CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    endif()
    if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
                CACHE PATH "Build directory for libraries")
        mark_as_advanced(CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    endif()
    if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
                CACHE PATH "Build directory for binaries")
        mark_as_advanced(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    endif()
    if(NOT CMAKE_PYTHON_OUTPUT_DIRECTORY)
        set(CMAKE_PYTHON_OUTPUT_DIRECTORY
            "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/site-packages"
            CACHE PATH "Build directory for python modules"
        )
    endif()
endmacro()


# set names and paths of install directories
# the defaults in CMake are sub-ideal for historic reasons, lets make them more
# Unix-ish and portable.
#
macro(set_default_install_dirs)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        include(GNUInstallDirs)
        if(NOT CMAKE_INSTALL_CMAKEDIR)
            set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake"
                    CACHE PATH "CMake config package location for installed targets")
            if(WIN32)
                set(CMAKE_INSTALL_LIBDIR Lib
                        CACHE PATH "Object code libraries")
                set_property(CACHE CMAKE_INSTALL_CMAKEDIR PROPERTY VALUE "cmake")
            endif()
            mark_as_advanced(CMAKE_INSTALL_CMAKEDIR)
        endif()
    endif()

    if(WIN32)
        set(FerroX_INSTALL_CMAKEDIR "${CMAKE_INSTALL_CMAKEDIR}")
    else()
        set(FerroX_INSTALL_CMAKEDIR "${CMAKE_INSTALL_CMAKEDIR}/FerroX")
    endif()
endmacro()


# set names and paths of install directories
# the defaults in CMake are sub-ideal for historic reasons, lets make them more
# Unix-ish and portable.
#
macro(ferrox_set_default_install_dirs)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        include(GNUInstallDirs)
        if(NOT CMAKE_INSTALL_CMAKEDIR)
            set(CMAKE_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake"
                    CACHE PATH "CMake config package location for installed targets")
            if(WIN32)
                set(CMAKE_INSTALL_LIBDIR Lib
                        CACHE PATH "Object code libraries")
                set_property(CACHE CMAKE_INSTALL_CMAKEDIR PROPERTY VALUE "cmake")
            endif()
            mark_as_advanced(CMAKE_INSTALL_CMAKEDIR)
        endif()
    endif()

    if(WIN32)
        set(FerroX_INSTALL_CMAKEDIR "${CMAKE_INSTALL_CMAKEDIR}")
    else()
        set(FerroX_INSTALL_CMAKEDIR "${CMAKE_INSTALL_CMAKEDIR}/FerroX")
    endif()
endmacro()


# set names and paths for Python modules
# this needs to be slightly delayed until we found Python and know its
# major and minor version number
#
macro(ferrox_set_default_install_dirs_python)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        # Python install and build output dirs
        if(NOT CMAKE_INSTALL_PYTHONDIR)
            if(WIN32)
                set(CMAKE_INSTALL_PYTHONDIR_DEFAULT
                        "${CMAKE_INSTALL_LIBDIR}/site-packages"
                        )
            else()
                set(CMAKE_INSTALL_PYTHONDIR_DEFAULT
                        "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages"
                        )
            endif()
            set(CMAKE_INSTALL_PYTHONDIR "${CMAKE_INSTALL_PYTHONDIR_DEFAULT}"
                    CACHE STRING "Location for installed python package"
                    )
        endif()
    endif()
endmacro()


# change the default CMAKE_BUILD_TYPE
# the default in CMake is Debug for historic reasons
#
macro(set_default_build_type default_build_type)
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
        set(CMAKE_CONFIGURATION_TYPES "Release;Debug;MinSizeRel;RelWithDebInfo")
        if(NOT CMAKE_BUILD_TYPE)
            set(CMAKE_BUILD_TYPE ${default_build_type}
                CACHE STRING
                "Choose the build type, e.g. Release, Debug, or RelWithDebInfo." FORCE)
            set_property(CACHE CMAKE_BUILD_TYPE
                PROPERTY STRINGS ${CMAKE_CONFIGURATION_TYPES})
        endif()
        if(NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
            message(WARNING "CMAKE_BUILD_TYPE '${CMAKE_BUILD_TYPE}' is not one of "
                    "${CMAKE_CONFIGURATION_TYPES}. Is this a typo?")
        endif()
    endif()
endmacro()

# Set CXX warning flags when FerroX_ENABLE_ALL_WARNINGS is enabled
# Based on ERF SetERFCompileFlags.cmake pattern
#
macro(set_cxx_warning_flags)
    if(FerroX_ENABLE_ALL_WARNINGS)
        # GCC, Clang, and Intel seem to accept these
        set(FerroX_CXX_FLAGS "-Wall" "-Wextra" "-pedantic")
        
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7.0)
            # Avoid notes about -faligned-new with GCC > 7
            list(APPEND FerroX_CXX_FLAGS "-faligned-new")
        endif()
        
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            # Intel always reports some diagnostics we don't necessarily care about
            list(APPEND FerroX_CXX_FLAGS "-diag-disable:11074,11076")
        endif()
        
        # Apply warning flags to all targets
        foreach(D IN LISTS FerroX_DIMS)
            ferrox_set_suffix_dims(SD ${D})
            if(FerroX_LIB)
                separate_arguments(FerroX_CXX_FLAGS)
                target_compile_options(lib_${SD} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${FerroX_CXX_FLAGS}>)
            endif()
        endforeach()
    endif()
endmacro()

# Set CXX
# Note: this is a bit legacy and one should use CMake TOOLCHAINS instead.
#
macro(set_cxx_warnings)
    # On Windows, Clang -Wall aliases -Weverything; default is /W3
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" AND NOT WIN32)
        # list(APPEND CMAKE_CXX_FLAGS "-fsanitize=address") # address, memory, undefined
        # set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=address")
        # set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
        # set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -fsanitize=address")

        # note: might still need a
        #   export LD_PRELOAD=libclang_rt.asan.so
        # or on Debian 9 with Clang 6.0
        #   export LD_PRELOAD=/usr/lib/llvm-6.0/lib/clang/6.0.0/lib/linux/libclang_rt.asan-x86_64.so:
        #                     /usr/lib/llvm-6.0/lib/clang/6.0.0/lib/linux/libclang_rt.ubsan_minimal-x86_64.so
        # at runtime when used with symbol-hidden code (e.g. pybind11 module)

        #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wshadow -Woverloaded-virtual -Wextra-semi -Wunreachable-code")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "AppleClang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wshadow -Woverloaded-virtual -Wextra-semi -Wunreachable-code")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wshadow -Woverloaded-virtual -Wunreachable-code")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        # Warning C4503: "decorated name length exceeded, name was truncated"
        # Symbols longer than 4096 chars are truncated (and hashed instead)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd4503")
        # Yes, you should build against the same C++ runtime and with same
        # configuration (Debug/Release). MSVC does inconvenient choices for their
        # developers, so be it. (Our Windows-users use conda-forge builds, which
        # are consistent.)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -wd4251")
    endif ()
endmacro()

# Enables interprocedural optimization for a list of targets
#
function(ferrox_enable_IPO all_targets_list)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT is_IPO_available)
    if(is_IPO_available)
        foreach(tgt IN ITEMS ${all_targets_list})
            set_target_properties(${tgt} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
        endforeach()
    else()
        message(FATAL_ERROR "Interprocedural optimization is not available, set FerroX_IPO=OFF")
    endif()
endfunction()

# Set the suffix for targets and binaries depending on dimension
#
# User specify 1;2;3;RZ;RCYLINDER;RSPHERE in FerroX_DIMS.
# We append to CMake targets and binaries the suffix "Nd" for 1,2,3 or otherwise the lowercase dimension string
#
macro(ferrox_set_suffix_dims suffix dim)
    if("${dim}" STREQUAL "RZ")
        set(${suffix} rz)
    elseif("${dim}" STREQUAL "RCYLINDER")
        set(${suffix} rcylinder)
    elseif("${dim}" STREQUAL "RSPHERE")
        set(${suffix} rsphere)
    else()
        set(${suffix} ${dim}d)
    endif()
endmacro()

# Take an <imported_target> and expose it as INTERFACE target with
# FerroX::thirdparty::<propagated_name> naming and SYSTEM includes.
#
function(ferrox_make_third_party_includes_system imported_target propagated_name)
    add_library(FerroX::thirdparty::${propagated_name} INTERFACE IMPORTED)
    target_link_libraries(FerroX::thirdparty::${propagated_name} INTERFACE ${imported_target})

    if(TARGET ${imported_target})
        get_target_property(imported_target_type ${imported_target} TYPE)
        if(NOT imported_target_type STREQUAL INTERFACE_LIBRARY)
            get_target_property(ALL_INCLUDES ${imported_target} INCLUDE_DIRECTORIES)
            if(ALL_INCLUDES)
                set_target_properties(FerroX::thirdparty::${propagated_name} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
                target_include_directories(FerroX::thirdparty::${propagated_name} SYSTEM INTERFACE ${ALL_INCLUDES})
            endif()
        endif()
    endif()
endfunction()


# Set a feature-based binary name for the FerroX executable following the same 
# naming logic as AMReX's GNU Make system
#
# Parameters:
#   D - dimension (e.g., 3)
#   SIMPLE_NAME - optional, if TRUE, uses just basic name without feature suffixes
#
function(ferrox_set_binary_name D)
    # Parse optional arguments
    set(options SIMPLE_NAME)
    set(oneValueArgs "")
    set(multiValueArgs "")
    cmake_parse_arguments(FERROX_NAME "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    ferrox_set_suffix_dims(SD ${D})

    set(ferrox_bin_names)
    if(FerroX_APP)
        list(APPEND ferrox_bin_names app_${SD})
        set_target_properties(app_${SD} PROPERTIES OUTPUT_NAME "main${SD}")
    endif()
    if(FerroX_LIB)
        list(APPEND ferrox_bin_names lib_${SD})
        # On WIN32, the OUTPUT_NAME must not collide between lib and app!
        if(WIN32)
            set_target_properties(lib_${SD} PROPERTIES OUTPUT_NAME "libferrox${SD}")
        else()
            set_target_properties(lib_${SD} PROPERTIES OUTPUT_NAME "ferrox${SD}")
        endif()
    endif()

    foreach(tgt IN LISTS ferrox_bin_names)
        # If SIMPLE_NAME is requested, skip all the feature suffixes
        if(FERROX_NAME_SIMPLE_NAME)
            set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".ex")
        else()
            # Machine suffix components in exact order from Make.defs:
            # $(lowercase_comp)$(archSuffix)$(PrecisionSuffix)$(DebugSuffix)$(ProfSuffix)$(MProfSuffix)$(MPISuffix)$(UPCXXSuffix)$(OMPSuffix)$(ACCSuffix)$(GPUSuffix)$(CUPTISuffix)$(USERSuffix)

            # lowercase_comp - compiler name
            if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".gnu")
            elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".intel")
            elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".intel-llvm")
            elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".llvm")
            elseif(CMAKE_CXX_COMPILER_ID STREQUAL "PGI" OR CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".nvhpc")
            elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Cray")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".cray")
            elseif(CMAKE_CXX_COMPILER_ID STREQUAL "XL")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".ibm")
            endif()

            # archSuffix - architecture suffix (typically from CRAY_CPU_TARGET)
            if(DEFINED ENV{CRAY_CPU_TARGET} AND NOT FerroX_COMPUTE STREQUAL "CUDA")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".$ENV{CRAY_CPU_TARGET}")
            endif()

            # PrecisionSuffix - precision
            if(FerroX_PRECISION STREQUAL "SINGLE")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".FLOAT")
            endif()

            # DebugSuffix - debug/test mode
            if(CMAKE_BUILD_TYPE MATCHES "Debug")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".DEBUG")
            elseif(FerroX_TESTING AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".TEST")
            endif()

            # ProfSuffix - profiling suffix
            if(AMReX_TRACE_PROFILE AND AMReX_COMM_PROFILE)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".COMTR_PROF")
            elseif(AMReX_TRACE_PROFILE AND NOT AMReX_COMM_PROFILE)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".TRACE_PROF")
            elseif(NOT AMReX_TRACE_PROFILE AND AMReX_COMM_PROFILE)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".COMM_PROF")
            elseif(NOT AMReX_TRACE_PROFILE AND NOT AMReX_COMM_PROFILE AND AMReX_BASE_PROFILE)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".PROF")
            elseif(AMReX_TINY_PROFILE)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".TPROF")
            endif()

            # MProfSuffix - memory profiling
            if(AMReX_MEM_PROFILE)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".MPROF")
            endif()

            # MPISuffix - MPI configuration
            if(FerroX_MPI_THREAD_MULTIPLE)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".MTMPI")
            elseif(FerroX_MPI)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".MPI")
            endif()


            # OMPSuffix - OpenMP configuration
            if(AMReX_OMP)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".OMP")
            endif()


            # GPUSuffix - GPU backend
            if(FerroX_COMPUTE STREQUAL "HIP")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".HIP")
            elseif(FerroX_COMPUTE STREQUAL "CUDA")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".CUDA")
            elseif(FerroX_COMPUTE STREQUAL "SYCL")
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".SYCL")
            endif()


            # FerroX-specific suffixes
            if(FerroX_TIME_DEPENDENT)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".TD")
            endif()

            if(FerroX_SUNDIALS)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".SUNDIALS")
            endif()

            if(FerroX_EB)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".EB")
            endif()

            # USERSuffix - user-defined suffix
            if(DEFINED FerroX_USER_SUFFIX)
                set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME "${FerroX_USER_SUFFIX}")
            endif()

            # Final .ex extension (matching GNU Make behavior)
            set_property(TARGET ${tgt} APPEND_STRING PROPERTY OUTPUT_NAME ".ex")
        endif()

        # Create symlinks for convenience
        if(FerroX_APP)
            # alias to the latest build, because using the full name is often confusing
            add_custom_command(TARGET app_${SD} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E create_symlink
                    $<TARGET_FILE_NAME:app_${SD}>
                    ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ferrox${SD}
            )
        endif()
        if(FerroX_LIB)
            add_custom_command(TARGET lib_${SD} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E create_symlink
                    $<TARGET_FILE_NAME:lib_${SD}>
                    $<TARGET_FILE_DIR:lib_${SD}>/libferrox${SD}$<TARGET_FILE_SUFFIX:lib_${SD}>
            )
        endif()
    endforeach()
endfunction()


# Set an MPI_TEST_EXE variable for test runs which runs num_ranks
# ranks. On some systems, you might need to use the a specific
# mpiexec wrapper, e.g. on Summit (ORNL) pass the hint
# -DMPIEXEC_EXECUTABLE=$(which jsrun) to run ctest.
#
function(configure_mpiexec num_ranks)
    # OpenMPI root guard: https://github.com/open-mpi/ompi/issues/4451
    if("$ENV{USER}" STREQUAL "root")
        # calling even --help as root will abort and warn on stderr
        execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --help
            ERROR_VARIABLE MPIEXEC_HELP_TEXT
            OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(${MPIEXEC_HELP_TEXT} MATCHES "^.*allow-run-as-root.*$")
                set(MPI_ALLOW_ROOT --allow-run-as-root)
            endif()
    endif()
    set(MPI_TEST_EXE
        ${MPIEXEC_EXECUTABLE}
        ${MPI_ALLOW_ROOT}
        ${MPIEXEC_NUMPROC_FLAG} ${num_ranks}
        PARENT_SCOPE
    )
endfunction()


# FUNCTION: get_source_version
#
# Retrieves source version info and sets internal cache variables
# ${NAME}_GIT_VERSION and ${NAME}_PKG_VERSION
#
function(get_source_version NAME SOURCE_DIR)
    find_package(Git QUIET)
    set(_tmp "")

    # Try to inquire software version from git
    if(EXISTS ${SOURCE_DIR}/.git AND ${GIT_FOUND})
        execute_process(COMMAND git describe --abbrev=12 --dirty --always --tags
            WORKING_DIRECTORY ${SOURCE_DIR}
            OUTPUT_VARIABLE _tmp)
        string( STRIP "${_tmp}" _tmp)
    endif()

    # Is there a CMake project version?
    # For deployed releases that build from tarballs, this is what we want to pick
    if(NOT _tmp AND ${NAME}_VERSION)
        set(_tmp "${${NAME}_VERSION}-nogit")
    endif()

    set(${NAME}_GIT_VERSION "${_tmp}" CACHE INTERNAL "")
    unset(_tmp)
endfunction ()


# Prints a summary of FerroX options at the end of the CMake configuration
#
function(ferrox_print_summary)
    message("")
    message("FerroX build configuration:")
    message("  Version: ${FerroX_VERSION} (${FerroX_GIT_VERSION})")
    message("  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} "
                            "${CMAKE_CXX_COMPILER_VERSION} "
                            "${CMAKE_CXX_COMPILER_WRAPPER}")
    message("    ${CMAKE_CXX_COMPILER}")
    message("")
    message("  Installation prefix: ${CMAKE_INSTALL_PREFIX}")
    message("        bin: ${CMAKE_INSTALL_BINDIR}")
    message("        lib: ${CMAKE_INSTALL_LIBDIR}")
    message("    include: ${CMAKE_INSTALL_INCLUDEDIR}")
    message("      cmake: ${FerroX_INSTALL_CMAKEDIR}")
    message("")
    set(BLD_TYPE_UNKNOWN "")
    if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR AND
       NOT CMAKE_BUILD_TYPE IN_LIST CMAKE_CONFIGURATION_TYPES)
        set(BLD_TYPE_UNKNOWN " (unknown type, check warning)")
    endif()
    message("  Build type: ${CMAKE_BUILD_TYPE}${BLD_TYPE_UNKNOWN}")
    set(LIB_TYPE "")
    if(FerroX_LIB)
        if(BUILD_SHARED_LIBS)
            set(LIB_TYPE " (shared")
        else()
            set(LIB_TYPE " (static")
        endif()
        if(FerroX_UNITY_BUILD)
            set(LIB_TYPE "${LIB_TYPE}, unity build")
        endif()
        set(LIB_TYPE "${LIB_TYPE})")
    endif()
    #message("  Testing: ${BUILD_TESTING}")
    message("  Build options:")
    message("    APP: ${FerroX_APP}")
    message("    COMPUTE: ${FerroX_COMPUTE}")
    message("    SIMD: ${FerroX_SIMD}")
    message("    DIMS: ${FerroX_DIMS}")
    message("    Embedded Boundary: ${FerroX_EB}")
    message("    IPO/LTO: ${FerroX_IPO}")
    message("    LIB: ${FerroX_LIB}${LIB_TYPE}")
    message("    MPI: ${FerroX_MPI}")
    if(MPI)
        message("    MPI (thread multiple): ${FerroX_MPI_THREAD_MULTIPLE}")
    endif()
    message("    PARTICLE PRECISION: ${FerroX_PARTICLE_PRECISION}")
    message("    PRECISION: ${FerroX_PRECISION}")
    message("")
endfunction()
