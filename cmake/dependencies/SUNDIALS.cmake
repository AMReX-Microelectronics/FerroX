# Create SUNDIALS target aliases for internal builds
# Maps internal SUNDIALS targets to standard SUNDIALS:: namespace
function(create_sundials_aliases)
    # Core components (always available)
    if(BUILD_SHARED_LIBS)
        if(NOT TARGET SUNDIALS::arkode)
            add_library(SUNDIALS::arkode ALIAS sundials_arkode_shared)
        endif()
        if(NOT TARGET SUNDIALS::cvode)
            add_library(SUNDIALS::cvode ALIAS sundials_cvode_shared)
        endif()
        if(NOT TARGET SUNDIALS::nvecserial)
            add_library(SUNDIALS::nvecserial ALIAS sundials_nvecserial_shared)
        endif()
        if(FerroX_MPI AND NOT TARGET SUNDIALS::nvecmpimanyvector)
            add_library(SUNDIALS::nvecmpimanyvector ALIAS sundials_nvecmpimanyvector_shared)
        endif()
    else()
        if(NOT TARGET SUNDIALS::arkode)
            add_library(SUNDIALS::arkode ALIAS sundials_arkode_static)
        endif()
        if(NOT TARGET SUNDIALS::cvode)
            add_library(SUNDIALS::cvode ALIAS sundials_cvode_static)
        endif()
        if(NOT TARGET SUNDIALS::nvecserial)
            add_library(SUNDIALS::nvecserial ALIAS sundials_nvecserial_static)
        endif()
        if(NOT TARGET SUNDIALS::nvecmanyvector)
            add_library(SUNDIALS::nvecmanyvector ALIAS sundials_nvecmanyvector_static)
        endif()
        if(FerroX_MPI AND NOT TARGET SUNDIALS::nvecmpimanyvector)
            add_library(SUNDIALS::nvecmpimanyvector ALIAS sundials_nvecmpimanyvector_static)
        endif()
    endif()
endfunction()

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

        #
        # To specify the location of a pre-installed SUNDIALS, set the
        # `SUNDIALS_DIR` variable or add the install directory to `CMAKE_PREFIX_PATH`.

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
            set(SUNDIALS_INDEX_SIZE 32 CACHE INTERNAL "")
            set(SUNDIALS_BUILD_PACKAGE_FUSED_KERNELS ON CACHE INTERNAL "")
            # Add CUDA-specific SUNDIALS options
            set(SUNDIALS_PRECISION "double" CACHE INTERNAL "")
            # Disable problematic CUDA components that require cusolver/cusparse
            set(BUILD_SUNMATRIX_CUSPARSE OFF CACHE INTERNAL "")
            set(BUILD_SUNLINSOL_CUSOLVERSP OFF CACHE INTERNAL "")
            set(ENABLE_CUSOLVER OFF CACHE INTERNAL "")
            set(ENABLE_CUSPARSE OFF CACHE INTERNAL "")
            set(ENABLE_CUBLAS OFF CACHE INTERNAL "")
            set(ENABLE_CURAND OFF CACHE INTERNAL "")
        elseif(FerroX_COMPUTE STREQUAL HIP)
            set(ENABLE_CUDA OFF CACHE INTERNAL "")
            set(ENABLE_HIP ON CACHE INTERNAL "")
            set(ENABLE_SYCL OFF CACHE INTERNAL "")
            set(SUNDIALS_BUILD_PACKAGE_FUSED_KERNELS ON CACHE INTERNAL "")
        elseif(FerroX_COMPUTE STREQUAL SYCL)
            set(ENABLE_CUDA OFF CACHE INTERNAL "")
            set(ENABLE_HIP OFF CACHE INTERNAL "")
            set(ENABLE_SYCL ON CACHE INTERNAL "")
            set(CMAKE_CXX_STANDARD 17 CACHE INTERNAL "")
        else()
            set(ENABLE_CUDA OFF CACHE INTERNAL "")
            set(ENABLE_HIP OFF CACHE INTERNAL "")
            set(ENABLE_SYCL OFF CACHE INTERNAL "")
        endif()

        # Precision settings for SUNDIALS (always use DOUBLE for compatibility)
        # SUNDIALS SINGLE precision has compatibility issues on Windows and HIP
        # Force DOUBLE precision for reliable cross-platform builds
        set(SUNDIALS_PRECISION "DOUBLE" CACHE INTERNAL "")
        
        # Note: To override this behavior and use SINGLE precision SUNDIALS
        # (not recommended), configure with:
        # cmake -DSUNDIALS_PRECISION=SINGLE ...

        # Enable required SUNDIALS components for FerroX
        set(ENABLE_ARKODE ON CACHE INTERNAL "")
        set(ENABLE_CVODE ON CACHE INTERNAL "")
        set(ENABLE_EXAMPLES OFF CACHE INTERNAL "")
        set(EXAMPLES_ENABLE_C OFF CACHE INTERNAL "")
        set(EXAMPLES_ENABLE_CUDA OFF CACHE INTERNAL "")
        set(ENABLE_UNIT_TESTS OFF CACHE INTERNAL "")

        # Library build configuration
        if(FerroX_LIB AND BUILD_SHARED_LIBS)
            set(BUILD_SHARED_LIBS ON CACHE INTERNAL "")
            set(SUNDIALS_BUILD_STATIC_LIBS OFF CACHE INTERNAL "")
        else()
            set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
            set(SUNDIALS_BUILD_STATIC_LIBS ON CACHE INTERNAL "")
        endif()

        # Position independent code for shared libraries
        if(ABLASTR_POSITION_INDEPENDENT_CODE OR (FerroX_LIB AND BUILD_SHARED_LIBS))
            set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE INTERNAL "")
        endif()

        # Install settings
        set(ENABLE_INSTALL_DOCS OFF CACHE INTERNAL "")

        if(FerroX_sundials_src)
            add_subdirectory(${FerroX_sundials_src} _deps/localsundials-build/)

        # For local source builds, set SUNDIALS_FOUND so AMReX knows it's available
        set(SUNDIALS_FOUND TRUE CACHE BOOL "SUNDIALS was built from local source" FORCE)
        else()
            FetchContent_Declare(fetchedsundials
                GIT_REPOSITORY ${FerroX_sundials_repo}
                GIT_TAG        ${FerroX_sundials_branch}
                BUILD_IN_SOURCE 0
            )
            FetchContent_MakeAvailable(fetchedsundials)

            # After FetchContent_MakeAvailable, the SUNDIALS targets are available
            # Set a variable so AMReX knows SUNDIALS is already available
            set(SUNDIALS_FOUND TRUE CACHE BOOL "SUNDIALS was built via FetchContent" FORCE)

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
        mark_as_advanced(SUNDIALS_INDEX_SIZE)
        mark_as_advanced(SUNDIALS_BUILD_PACKAGE_FUSED_KERNELS)

        # Extract SUNDIALS version from its own config files
        if(FerroX_sundials_src)
            # For local source builds
            if(EXISTS "${CMAKE_BINARY_DIR}/_deps/localsundials-build/SUNDIALSConfigVersion.cmake")
                include("${CMAKE_BINARY_DIR}/_deps/localsundials-build/SUNDIALSConfigVersion.cmake")
                set(SUNDIALS_VERSION "${PACKAGE_VERSION}" CACHE STRING "SUNDIALS version from local build" FORCE)
            endif()
        else()
            # For FetchContent builds - use FetchContent variables
            FetchContent_GetProperties(fetchedsundials)
            if(EXISTS "${fetchedsundials_BINARY_DIR}/SUNDIALSConfigVersion.cmake")
                include("${fetchedsundials_BINARY_DIR}/SUNDIALSConfigVersion.cmake")
                set(SUNDIALS_VERSION "${PACKAGE_VERSION}" CACHE STRING "SUNDIALS version from FetchContent" FORCE)
            endif()
        endif()

        # Fallback if version file not found
        if(NOT DEFINED SUNDIALS_VERSION)
            set(SUNDIALS_VERSION "7.0.0" CACHE STRING "SUNDIALS version (fallback)" FORCE)
        endif()

        message(STATUS "SUNDIALS: Using internal build (version ${SUNDIALS_VERSION})")

        # Create standard SUNDIALS:: aliases for internal build targets
        create_sundials_aliases()
    else()
        message(STATUS "Searching for pre-installed SUNDIALS ...")
        
        if (SUNDIALS_FOUND)
            message(STATUS "SUNDIALS_FOUND is true, using pre-configured SUNDIALS for version 6.0.0 or higher")
        else()
            set(SUNDIALS_MINIMUM_VERSION 6.0.0 CACHE INTERNAL "Minimum required SUNDIALS version")
            set(SUNDIALS_COMPONENTS 
                arkode cvode 
                nvecserial nvecmanyvector nvecmpimanyvector
                sunlinsolspgmr sunlinsolspfgmr sunnonlinsolfixedpoint)

            find_package(SUNDIALS CONFIG REQUIRED
                         COMPONENTS ${SUNDIALS_COMPONENTS}
                         OPTIONAL_COMPONENTS core
                         PATHS ${SUNDIALS_ROOT} $ENV{SUNDIALS_ROOT})

            if(SUNDIALS_VERSION VERSION_LESS ${SUNDIALS_MINIMUM_VERSION})
                message(FATAL_ERROR "SUNDIALS_VERSION ${SUNDIALS_MINIMUM_VERSION} or newer is required. Found version ${SUNDIALS_VERSION}.")
            endif()

            message(STATUS "SUNDIALS: Found version '${SUNDIALS_VERSION}'")
        endif()
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

set(FerroX_sundials_branch "main"
    CACHE STRING
    "Repository branch for FerroX_sundials_repo if(FerroX_sundials_internal)")

# Internal build option - matches AMReX pattern
option(FerroX_sundials_internal "Download & build SUNDIALS" ON)

# Call the macro
if(FerroX_SUNDIALS)
    message(STATUS "Calling find_sundials here")
    find_sundials()
endif()
