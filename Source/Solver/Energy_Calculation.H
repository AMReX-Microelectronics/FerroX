#ifndef FERROX_TOTALFREEENERGY_H_
#define FERROX_TOTALFREEENERGY_H_

#include <AMReX_Array.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

// Computes the Landau free energy over the ferroelectric region (MaterialMask == 0.0)
Real ComputeLandauEnergy( Array<MultiFab, AMREX_SPACEDIM>& P,
                          MultiFab& MaterialMask,
                         const Geometry& geom,
                         Real alpha,
                         Real beta,
                         Real gamma_coeff);


Real ComputeElectrostaticEnergy( Array<MultiFab, AMREX_SPACEDIM>& P,
                                 Array<MultiFab, AMREX_SPACEDIM>& E,
                                const MultiFab& MaterialMask,
                                const Geometry& geom);


Real ComputeGradientEnergy(const Array<MultiFab, AMREX_SPACEDIM>& P,
                            const MultiFab& MaterialMask,
                            const Geometry& geom,
                            Real g11,
                            Real g44);
#endif