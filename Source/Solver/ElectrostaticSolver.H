#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include "FerroX.H"
#include "Input/BoundaryConditions/BoundaryConditions.H"
#include "Input/GeometryProperties/GeometryProperties.H"

using namespace amrex;
using namespace FerroX;

void ComputePoissonRHS(MultiFab&               PoissonRHS, 
		Array<MultiFab, AMREX_SPACEDIM> &P_old,
		MultiFab&                      rho, 
		MultiFab&                      MaterialMask, 
		const Geometry&                 geom);

//void ComputeEfromPhi(MultiFab&                 PoissonPhi,
//                MultiFab&                      Ex,
//                MultiFab&                      Ey,
//                MultiFab&                      Ez,
//                const Geometry&                 geom);
//
void ComputeEfromPhi(MultiFab&                 PoissonPhi,
                MultiFab&                      Ex,
                MultiFab&                      Ey,
                MultiFab&                      Ez,
                const Geometry&                 geom,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi);

//mask based permittivity
void InitializePermittivity(std::array<std::array<amrex::LinOpBCType,AMREX_SPACEDIM>,2>& LinOpBCType_2d, MultiFab& beta_cc, const MultiFab& MaterialMask, const amrex::GpuArray<int, AMREX_SPACEDIM>& n_cell, const Geometry& geom);

void dF_dPhi(MultiFab&            alpha_cc,
             MultiFab&            PoissonRHS, 
             MultiFab&            PoissonPhi, 
	     Array<MultiFab, AMREX_SPACEDIM>& P_old,
             MultiFab&            rho,
             MultiFab&            e_den,
             MultiFab&            p_den,
	     MultiFab&      MaterialMask,
             const          Geometry& geom,
	     const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi);

void ComputePoissonRHS_Newton(MultiFab& PoissonRHS, 
                              MultiFab& PoissonPhi, 
                              MultiFab& alpha_cc);

void SetPhiBC_z(MultiFab& PossonPhi, const amrex::GpuArray<int, AMREX_SPACEDIM>& n_cell);

void SetPoissonBC(c_FerroX& rFerroX, std::array<std::array<amrex::LinOpBCType,AMREX_SPACEDIM>,2>& LinOpBCType_2d, bool& all_homogeneous_boundaries, bool& some_functionbased_inhomogeneous_boundaries, bool& some_constant_inhomogeneous_boundaries);

void Fill_Constant_Inhomogeneous_Boundaries(c_FerroX& rFerroX, MultiFab& PoissonPhi);
void Fill_FunctionBased_Inhomogeneous_Boundaries(c_FerroX& rFerroX, MultiFab& PoissonPhi, amrex::Real& time);

