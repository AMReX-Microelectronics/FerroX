# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../amrex
CODE_HOME := ..

DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = TRUE
USE_CUDA     = TRUE
COMP         = gnu
DIM          = 3
CXXSTD       = c++17

USE_EB = FALSE

ifeq ($(USE_CUDA), TRUE)
 USE_OMP= FALSE
endif

PRINT_NAME   = FALSE
PRINT_LOW   = FALSE
PRINT_HIGH   = FALSE
TIME_DEPENDENT = FALSE

include $(CODE_HOME)/Source/Make.FerroX

