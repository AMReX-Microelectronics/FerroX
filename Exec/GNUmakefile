# AMREX_HOME defines the directory in which we will find all the AMReX code.
AMREX_HOME ?= ../../amrex
CODE_HOME := ..

DEBUG        = FALSE
USE_MPI      = TRUE
USE_OMP      = TRUE
USE_CUDA     = TRUE
COMP         = gnu
DIM          = 3
TINY_PROFILE = TRUE

USE_EB = FALSE
USE_SUNDIALS = FALSE
SUNDIALS_HOME ?= ../../sundials/instdir

ifeq ($(USE_CUDA), TRUE)
 USE_OMP= FALSE
endif

PRINT_NAME   = FALSE
PRINT_LOW   = FALSE
PRINT_HIGH   = FALSE
TIME_DEPENDENT = FALSE

include $(CODE_HOME)/Source/Make.FerroX
