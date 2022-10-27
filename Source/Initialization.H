#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void InitializePandRho(int prob_type,
                   Array<MultiFab, AMREX_SPACEDIM> &P_old,
                   MultiFab&   Gamma,
                   MultiFab&   rho,
                   MultiFab&   e_den,
                   MultiFab&   p_den,
                   Real        SC_lo,
                   Real        SC_hi,
                   Real        DE_lo,
                   Real        DE_hi,
                   Real        BigGamma,
                   Real        q,
                   Real        Ec,
                   Real        Ev,
                   Real        kb,
                   Real        T,
                   Real        Nc,
                   Real        Nv,
                   amrex::GpuArray<amrex::Real, 3> prob_lo,
                   amrex::GpuArray<amrex::Real, 3> prob_hi,
                   const       Geometry& geom);

