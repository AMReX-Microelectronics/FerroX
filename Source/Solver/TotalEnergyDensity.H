#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include "FerroX.H"

using namespace amrex;
using namespace FerroX;

void CalculateTDGL_RHS(Array<MultiFab, AMREX_SPACEDIM> &GL_rhs,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                Array<MultiFab, AMREX_SPACEDIM> &E,
                MultiFab&                       Gamma,
                MultiFab&                       MaterialMask,
                MultiFab&                       tphaseMask,
                MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta,
                const Geometry& geom,
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi);

void Compute_P_Sum(const std::array<MultiFab, AMREX_SPACEDIM>& P, Real& sum);

void Compute_P_index_Sum(const MultiFab& MaterialMask, Real& count);

void Compute_P_av(const std::array<MultiFab, AMREX_SPACEDIM>& P, Real& sum, const MultiFab& MaterialMask, Real& count, Real &P_av_z);

