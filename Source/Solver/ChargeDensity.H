#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include "FerroX.H"

using namespace amrex;
using namespace FerroX;


void ComputeRho(MultiFab&      PoissonPhi,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                MultiFab&      rho,
                MultiFab&      e_den,
                MultiFab&      p_den,
                const MultiFab& MaterialMask,
                const Geometry& geom,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi);

void Compute_P_Sum(const std::array<MultiFab, AMREX_SPACEDIM>& P, Real& sum);

void Compute_P_index_Sum(const MultiFab& MaterialMask, Real& count);

void Compute_P_av(const std::array<MultiFab, AMREX_SPACEDIM>& P, Real& sum, const MultiFab& MaterialMask, Real& count, Real &P_av_z);
