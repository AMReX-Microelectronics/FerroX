#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include "FerroX.H"
#include "Input/GeometryProperties/GeometryProperties.H"

using namespace amrex;
using namespace FerroX;

void InitializePandRho(Array<MultiFab, AMREX_SPACEDIM> &P_old,
                   MultiFab&   Gamma,
                   MultiFab&   rho,
                   MultiFab&   e_den,
                   MultiFab&   p_den,
		   const MultiFab& MaterialMask,
		   const MultiFab& tphaseMask,
                   const amrex::GpuArray<int, AMREX_SPACEDIM>& n_cell,
                   const       Geometry& geom,
		   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi);

void InitializeMaterialMask(MultiFab& MaterialMask,
                            const Geometry& geom,
                            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi);

void InitializeMaterialMask(c_FerroX& rFerroX, const Geometry& geom, MultiFab& MaterialMask);
void Initialize_tphase_Mask(c_FerroX& rFerroX, const Geometry& geom, MultiFab& tphaseMask);
void Initialize_Euler_angles(c_FerroX& rFerroX, const Geometry& geom, MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta);
void Initialize_nodal_Euler_angles(c_FerroX& rFerroX, const Geometry& geom, MultiFab& Nodal_angle_alpha, MultiFab& Nodal_angle_beta, MultiFab& Nodal_angle_theta);

