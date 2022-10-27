#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void ComputePoissonRHS(MultiFab&               PoissonRHS, 
		Array<MultiFab, AMREX_SPACEDIM> &P_old,
		MultiFab&                      rho, 
		Real                           FE_lo, 
		Real                           FE_hi, 
		Real                           DE_lo, 
		Real                           DE_hi, 
		Real                           SC_lo, 
		Real                           SC_hi, 
		int                            P_BC_flag_lo, 
		int                            P_BC_flag_hi, 
		Real                           lambda, 
		amrex::GpuArray<amrex::Real, 3> prob_lo, 
		amrex::GpuArray<amrex::Real, 3> prob_hi, 
		const Geometry&                 geom);

void ComputeEfromPhi(MultiFab&                 PoissonPhi,
                MultiFab&                      Ex,
                MultiFab&                      Ey,
                MultiFab&                      Ez,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
		amrex::Real const Phi_Bc_hi,
                amrex::Real const Phi_Bc_lo,
                const Geometry&                 geom);

void InitializePermittivity(std::array< MultiFab, AMREX_SPACEDIM >& beta_face,
                Real                            FE_lo,
                Real                            FE_hi,
                Real                            DE_lo,
                Real                            DE_hi,
                Real                            SC_lo,
                Real                            SC_hi,
                Real                            epsilon_0,
                Real                            epsilonX_fe,
                Real                            epsilonZ_fe,
                Real                            epsilon_de,
                Real                            epsilon_si,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
                const Geometry&                 geom);

void SetPhiBC_z(MultiFab& PossonPhi,
                amrex::GpuArray<int, 3> n_cell,
                Real                    Phi_Bc_lo,
                Real                    Phi_Bc_hi);

void dF_dPhi(MultiFab&            alpha_cc,
             MultiFab&            PoissonRHS, 
             MultiFab&            PoissonPhi, 
             Real                 delta,
	     Array<MultiFab, AMREX_SPACEDIM>& P_old,
             MultiFab&            rho,
             MultiFab&            e_den,
             MultiFab&            p_den,
             Real                 FE_lo,
             Real                 FE_hi,
             Real                 DE_lo,
             Real                 DE_hi,
             Real                 Sc_lo,
             Real                 SC_hi,
             Real                 q,  
             Real                 Ec, 
             Real                 Ev, 
             Real                 kb, 
             Real                 T,  
             Real                 Nc, 
             Real                 Nv, 
             int                 P_BC_flag_lo,
             int                 P_BC_flag_hi,
             Real                lambda,
             amrex::GpuArray<amrex::Real, 3> prob_lo,
             amrex::GpuArray<amrex::Real, 3> prob_hi,
             const          Geometry& geom);

void ComputePoissonRHS_Newton(MultiFab& PoissonRHS, 
                              MultiFab& PoissonPhi, 
                              MultiFab& alpha_cc);

