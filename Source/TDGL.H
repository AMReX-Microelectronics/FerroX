#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

void InitializeRho(MultiFab& rho, MultiFab& e_den, MultiFab& p_den, Real Sc_lo, Real SC_hi, Real q, Real Ec, Real Ev, Real kb, Real T, Real Nc, Real Nv, const Geometry& geom);
void ComputeRho(MultiFab& PoissonPhi, MultiFab& rho, MultiFab& e_den, MultiFab& p_den, Real Sc_lo, Real SC_hi, Real q, Real Ec, Real Ev, Real kb, Real T, Real Nc, Real Nv, const Geometry& geom);
void ComputePoissonRHS(MultiFab& PoissonRHS, MultiFab& P_old, MultiFab& rho, Real FE_lo, Real FE_hi, Real DE_lo, Real DE_hi, Real SC_lo, Real SC_hi, int P_BC_flag_lo, int P_BC_flag_hi, Real lambda, amrex::GpuArray<amrex::Real, 3> prob_lo, amrex::GpuArray<amrex::Real, 3> prob_hi, const Geometry& geom);
void CalculateTDGL_RHS(MultiFab& GL_rhs, MultiFab& P_old, MultiFab& PoissonPhi, MultiFab& Gamma, Real FE_lo, Real FE_hi, Real DE_lo, Real DE_hi, Real SC_lo, Real SC_hi, int P_BC_flag_lo, int P_BC_flag_hi, int Phi_Bc_lo, int Phi_Bc_hi, Real alpha, Real beta, Real gamma, Real g11, Real g44, Real lambda, amrex::GpuArray<amrex::Real, 3> prob_lo, amrex::GpuArray<amrex::Real, 3> prob_hi, const Geometry& geom);
