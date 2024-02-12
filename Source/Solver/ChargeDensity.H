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

void Compute_Current(MultiFab&      PoissonPhi,
                     MultiFab&      e_den,
                     MultiFab&      p_den,
                     Array<MultiFab, AMREX_SPACEDIM> &Jn,
                     Array<MultiFab, AMREX_SPACEDIM> &Jp,
                     MultiFab& MaterialMask,
                     const Geometry& geom);

void Compute_Effective_Potentials(const MultiFab& PoissonPhi,
                                        MultiFab& e_potential,
                                        MultiFab& p_potential,
                                        const Geometry& geom);

void ComputeRho_DriftDiffusion(MultiFab&      PoissonPhi,
                Array<MultiFab, AMREX_SPACEDIM> &Jn,
                Array<MultiFab, AMREX_SPACEDIM> &Jp,
                MultiFab&      rho,
                MultiFab&      e_den,
                MultiFab&      p_den,
                MultiFab&      e_den_old,
                MultiFab&      p_den_old,
                MultiFab& MaterialMask,
                const Geometry& geom);
