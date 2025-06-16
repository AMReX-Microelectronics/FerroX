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

void CalculateDriftDiffusionCurrents(
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Jn, // OUT: Electron current density components (x,y,z)
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Jp, // OUT: Hole current density components (x,y,z)
    const amrex::MultiFab& e_den,                      // IN: Electron density
    const amrex::MultiFab& p_den,                      // IN: Hole density
    const amrex::MultiFab& PoissonPhi,                 // IN: Electric potential
    const amrex::Geometry& geom                        // IN: Simulation geometry
);
/*
void Compute_Effective_Potentials(const MultiFab& PoissonPhi,
                                        MultiFab& e_potential,
                                        MultiFab& p_potential,
                                        const Geometry& geom);
*/

void Compute_Effective_Potentials(const MultiFab& PoissonPhi,
                                  const MultiFab& e_den,
                                  const MultiFab& p_den,
                                  MultiFab& e_potential,
                                  MultiFab& p_potential,
                                  const Geometry& geom);

void ComputeRho_DriftDiffusion(MultiFab&      PoissonPhi,
                MultiFab&      rho,
                Array<MultiFab, AMREX_SPACEDIM> &Jn,
                Array<MultiFab, AMREX_SPACEDIM> &Jp,
                MultiFab&      e_den,
                MultiFab&      p_den,
                MultiFab&      e_den_old,
                MultiFab&      p_den_old,
                MultiFab& MaterialMask,
                const Geometry& geom);
