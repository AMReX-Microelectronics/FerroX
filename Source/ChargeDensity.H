#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void ComputeRho(MultiFab&      PoissonPhi,
                MultiFab&      rho,
                MultiFab&      e_den,
                MultiFab&      p_den,
                Real           Sc_lo,
                Real           SC_hi,
                Real           q, Real Ec,
                Real           Ev,
                Real           kb,
                Real           T,
                Real           Nc,
                Real           Nv,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
                const          Geometry& geom);

