#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include "FerroX.H"

using namespace amrex;
using namespace FerroX;

void ComputeRho(MultiFab&      PoissonPhi,
                MultiFab&      rho,
                MultiFab&      e_den,
                MultiFab&      p_den,
                const MultiFab& MaterialMask);

void ComputeRho_DriftDiffusion(MultiFab&      PoissonPhi,
                               Array<MultiFab, AMREX_SPACEDIM> &Jn,
                               Array<MultiFab, AMREX_SPACEDIM> &Jp,
                               MultiFab&      rho,
                               MultiFab&      e_den,
                               MultiFab&      p_den,
                               const Geometry& geom,
                               const MultiFab& MaterialMask);
