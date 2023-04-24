#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include "FerroX.H"

using namespace amrex;
using namespace FerroX;

void transform_global_to_local(Array<MultiFab, AMREX_SPACEDIM> &src,
		               Array<MultiFab, AMREX_SPACEDIM> &dst,
			       MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta);

void transform_local_to_global(Array<MultiFab, AMREX_SPACEDIM> &src,
		               Array<MultiFab, AMREX_SPACEDIM> &dst,
			       MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta);
