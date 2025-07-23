macro(find_sundials)
    if(FerroX_sundials_src)
        message(STATUS "Compiling local SUNDIALS ...")
        message(STATUS "SUNDIALS source path: ${FerroX_sundials_src}")
        if(NOT IS_DIRECTORY ${FerroX_sundials_src})
            message(FATAL_ERROR "Specified directory FerroX_sundials_src='${FerroX_sundials_src}' does not exist!")
        endif()
    elseif(FerroX_sundials_internal)
        message(STATUS "Downloading SUNDIALS ...")
        message(STATUS "SUNDIALS repository: ${FerroX_sundials_repo} (${FerroX_sundials_branch})")
        include(FetchContent)
    endif()

    if(FerroX_sundials_internal OR FerroX_sundials_src)
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

        # Configure SUNDIALS to match AMReX settings
        # See https://sundials.readthedocs.io/en/latest/Installation.html#configuration-options

        # Enable/disable MPI support to match AMReX
        if(FerroX_MPI)
            set(ENABLE_MPI ON CACHE INTERNAL "")
        else()
            set(ENABLE_MPI OFF CACHE INTERNAL "")
        endif()

        # Enable/disable OpenMP support to match AMReX
        if(FerroX_COMPUTE STREQUAL OMP)
            set(ENABLE_OPENMP ON CACHE INTERNAL "")
        else()
            set(ENABLE_OPENMP OFF CACHE INTERNAL "")
        endif()

        # Enable/disable GPU support to match AMReX
        if(FerroX_COMPUTE STREQUAL CUDA)
            set(ENABLE_CUDA ON CACHE INTERNAL "")
            set(ENABLE_HIP OFF CACHE INTERNAL "")
            set(ENABLE_SYCL OFF CACHE INTERNAL "")
        elseif(FerroX_COMPUTE STREQUAL HIP)
            set(ENABLE_CUDA OFF CACHE INTERNAL "")
            set(ENABLE_HIP ON CACHE INTERNAL "")
            set(ENABLE_SYCL OFF CACHE INTERNAL "")
        elseif(FerroX_COMPUTE STREQUAL SYCL)
            set(ENABLE_CUDA OFF CACHE INTERNAL "")
            set(ENABLE_HIP OFF CACHE INTERNAL "")
            set(ENABLE_SYCL ON CACHE INTERNAL "")
        else()
            set(ENABLE_CUDA OFF CACHE INTERNAL "")
            set(ENABLE_HIP OFF CACHE INTERNAL "")
            set(ENABLE_SYCL OFF CACHE INTERNAL "")
        endif()

        # Precision settings to match AMReX
        if(FerroX_PRECISION STREQUAL "DOUBLE")
            set(SUNDIALS_PRECISION "DOUBLE" CACHE INTERNAL "")
        else()
            set(SUNDIALS_PRECISION "SINGLE" CACHE INTERNAL "")
        endif()

        # Enable required SUNDIALS components for FerroX
        set(ENABLE_ARKODE ON CACHE INTERNAL "")
        set(ENABLE_CVODE ON CACHE INTERNAL "")
        set(ENABLE_EXAMPLES OFF CACHE INTERNAL "")
        set(ENABLE_UNIT_TESTS OFF CACHE INTERNAL "")

        # Shared library settings
        if(FerroX_PYTHON OR (FerroX_LIB AND BUILD_SHARED_LIBS))
            set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
            set(SUNDIALS_BUILD_STATIC_LIBS OFF CACHE INTERNAL "")
        else()
            set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
            set(SUNDIALS_BUILD_STATIC_LIBS ON CACHE INTERNAL "")
        endif()

        # Position independent code for shared libraries
        if(FerroX_PYTHON OR 
           ABLASTR_POSITION_INDEPENDENT_CODE OR
           (FerroX_LIB AND BUILD_SHARED_LIBS))
            set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE INTERNAL "")
        endif()

        # Install settings
        set(ENABLE_INSTALL_DOCS OFF CACHE INTERNAL "")

        if(FerroX_sundials_src)
            add_subdirectory(${FerroX_sundials_src} _deps/localsundials-build/)
        else()
            FetchContent_Declare(fetchedsundials
                GIT_REPOSITORY ${FerroX_sundials_repo}
                GIT_TAG        ${FerroX_sundials_branch}
                BUILD_IN_SOURCE 0
            )
            FetchContent_MakeAvailable(fetchedsundials)

            # Advanced fetch options
            mark_as_advanced(FETCHCONTENT_SOURCE_DIR_FETCHEDSUNDIALS)
            mark_as_advanced(FETCHCONTENT_UPDATES_DISCONNECTED_FETCHEDSUNDIALS)
        endif()

        # Mark advanced options to keep the UI clean
        mark_as_advanced(ENABLE_ARKODE)
        mark_as_advanced(ENABLE_CVODE)
        mark_as_advanced(ENABLE_EXAMPLES)
        mark_as_advanced(ENABLE_UNIT_TESTS)
        mark_as_advanced(ENABLE_MPI)
        mark_as_advanced(ENABLE_OPENMP)
        mark_as_advanced(ENABLE_CUDA)
        mark_as_advanced(ENABLE_HIP)
        mark_as_advanced(ENABLE_SYCL)
        mark_as_advanced(SUNDIALS_PRECISION)
        mark_as_advanced(BUILD_SHARED_LIBS)
        mark_as_advanced(SUNDIALS_BUILD_STATIC_LIBS)
        mark_as_advanced(ENABLE_INSTALL_DOCS)

        message(STATUS "SUNDIALS: Using internal build")
    else()
        message(STATUS "Searching for pre-installed SUNDIALS ...")
        
        # Use the minimum version required by AMReX (6.0.0+)
        set(SUNDIALS_MINIMUM_VERSION 6.0.0)
        set(SUNDIALS_COMPONENTS arkode cvode sunlinsolspgmr sunlinsolspfgmr
            sunlinsolsptfqmr sunnonlinsolnewton sunlinsolklu sunlinsollapackband
            sunlinsollapackdense nvecserial sunmatrixband sunmatrixdense
            sunmatrixsparse)

        find_package(SUNDIALS CONFIG REQUIRED
                     COMPONENTS ${SUNDIALS_COMPONENTS}
                     PATHS ${SUNDIALS_ROOT} $ENV{SUNDIALS_ROOT})

        if(SUNDIALS_VERSION VERSION_LESS ${SUNDIALS_MINIMUM_VERSION})
            message(FATAL_ERROR "SUNDIALS_VERSION ${SUNDIALS_MINIMUM_VERSION} or newer is required. Found version ${SUNDIALS_VERSION}.")
        endif()

        message(STATUS "SUNDIALS: Found version '${SUNDIALS_VERSION}'")
    endif()
endmacro()

# Local source-tree option 
set(FerroX_sundials_src ""
    CACHE PATH
    "Local path to SUNDIALS source directory (preferred if set)")

# Git fetcher options
set(FerroX_sundials_repo "https://github.com/LLNL/sundials.git"
    CACHE STRING
    "Repository URI to pull and build SUNDIALS from if(FerroX_sundials_internal)")

set(FerroX_sundials_branch "release"
    CACHE STRING
    "Repository branch for FerroX_sundials_repo if(FerroX_sundials_internal)")

# Internal build option - matches AMReX pattern
option(FerroX_sundials_internal "Download & build SUNDIALS" ON)

# Call the macro
if(FerroX_SUNDIALS)
    find_sundials()
endif()