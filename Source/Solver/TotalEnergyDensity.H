#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include "FerroX.H"

using namespace amrex;
using namespace FerroX;

void CalculateTDGL_RHS(Array<MultiFab, AMREX_SPACEDIM> &GL_rhs,
                Array<MultiFab, AMREX_SPACEDIM> &GL_rhs_Landau,
                Array<MultiFab, AMREX_SPACEDIM> &GL_rhs_grad,
                Array<MultiFab, AMREX_SPACEDIM> &GL_rhs_elec,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                Array<MultiFab, AMREX_SPACEDIM> &E, 
                MultiFab&                       Gamma,
                MultiFab&                 MaterialMask,
                MultiFab&                 tphaseMask,
                MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta,
                const Geometry& geom);

void Calculate_Landau(Array<MultiFab, AMREX_SPACEDIM> &GL_rhs_Landau,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                MultiFab&                       Gamma,
                MultiFab&                 tphaseMask);

void Calculate_Grad(Array<MultiFab, AMREX_SPACEDIM> &GL_rhs_grad,
                Array<MultiFab, AMREX_SPACEDIM> &P_old, 
                MultiFab&                       Gamma,
                MultiFab&                 MaterialMask,
                MultiFab&                 tphaseMask,
                MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta,
                const Geometry& geom);

void Calculate_Elec(Array<MultiFab, AMREX_SPACEDIM> &GL_rhs_elec,
                Array<MultiFab, AMREX_SPACEDIM> &E, 
                MultiFab&                       Gamma,
                MultiFab&                 tphaseMask);


