#ifndef _ferrox_H_
#define _ferrox_H_

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

using namespace amrex;

class FerroX 
{
public:

    FerroX ();
    ~FerroX ();

    static int max_grid_size;
    static int nsteps;
    static int plot_int;

    // time step
    static amrex::Real dt;

    static amrex::Vector<int> bc_lo;
    static amrex::Vector<int> bc_hi;

    static amrex::GpuArray<int, AMREX_SPACEDIM> n_cell; // number of cells in each direction
    static amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo; // physical lo coordinate
    static amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi; // physical hi coordinate

    // multimaterial stack geometry
    static amrex::Real DE_lo;
    static amrex::Real FE_lo;
    static amrex::Real SC_lo;
    static amrex::Real DE_hi;
    static amrex::Real FE_hi;
    static amrex::Real SC_hi;

    // material parameters
    static amrex::Real epsilon_0;
    static amrex::Real epsilonX_fe;
    static amrex::Real epsilonZ_fe;
    static amrex::Real epsilon_de;
    static amrex::Real epsilon_si;
    static amrex::Real alpha; // alpha = 2*alpha_1
    static amrex::Real beta; // beta = 4*alpha_11
    static amrex::Real gamma; // gamma = 6*alpha_111
    static amrex::Real BigGamma;
    static amrex::Real g11;
    static amrex::Real g44;
    static amrex::Real g44_p;
    static amrex::Real g12;
    static amrex::Real alpha_12;
    static amrex::Real alpha_112;
    static amrex::Real alpha_123;

    // Constants for SC layer calculations
    static amrex::Real Nc;
    static amrex::Real Nv;
    static amrex::Real Ec;
    static amrex::Real Ev;
    static amrex::Real q;
    static amrex::Real kb;
    static amrex::Real T;

    // P and Phi Bc
    static int P_BC_flag_lo;
    static int P_BC_flag_hi;
    static amrex::Real lambda;
    static amrex::Real Phi_Bc_lo;
    static amrex::Real Phi_Bc_hi;
    static amrex::Real Phi_Bc_inc;
    static int inc_step;

    //problem type : initialization of P for 2D/3D/convergence problems
    static int prob_type;

    static int mlmg_verbosity;

    static int TimeIntegratorOrder;

    static amrex::Real delta;

};

    void main_main ();
    void InitializeGlobalVariables ();
#endif
