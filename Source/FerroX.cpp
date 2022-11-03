#include "FerroX.H"

#include <AMReX_ParmParse.H>

AMREX_GPU_MANAGED int FerroX::max_grid_size;
AMREX_GPU_MANAGED int FerroX::nsteps;
AMREX_GPU_MANAGED int FerroX::plot_int;

// time step
AMREX_GPU_MANAGED amrex::Real FerroX::dt;

amrex::Vector<int> FerroX::bc_lo;
amrex::Vector<int> FerroX::bc_hi;

AMREX_GPU_MANAGED amrex::GpuArray<int, AMREX_SPACEDIM> FerroX::n_cell; // number of cells in each direction
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> FerroX::prob_lo; // physical lo coordinate
AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> FerroX::prob_hi; // physical hi coordinate

// multimaterial stack geometry
AMREX_GPU_MANAGED amrex::Real FerroX::DE_lo;
AMREX_GPU_MANAGED amrex::Real FerroX::FE_lo;
AMREX_GPU_MANAGED amrex::Real FerroX::SC_lo;
AMREX_GPU_MANAGED amrex::Real FerroX::DE_hi;
AMREX_GPU_MANAGED amrex::Real FerroX::FE_hi;
AMREX_GPU_MANAGED amrex::Real FerroX::SC_hi;

// material parameters
AMREX_GPU_MANAGED amrex::Real FerroX::epsilon_0;
AMREX_GPU_MANAGED amrex::Real FerroX::epsilonX_fe;
AMREX_GPU_MANAGED amrex::Real FerroX::epsilonZ_fe;
AMREX_GPU_MANAGED amrex::Real FerroX::epsilon_de;
AMREX_GPU_MANAGED amrex::Real FerroX::epsilon_si;
AMREX_GPU_MANAGED amrex::Real FerroX::alpha; // alpha = 2*alpha_1
AMREX_GPU_MANAGED amrex::Real FerroX::beta; // beta = 4*alpha_11
AMREX_GPU_MANAGED amrex::Real FerroX::gamma; // gamma = 6*alpha_111
AMREX_GPU_MANAGED amrex::Real FerroX::BigGamma;
AMREX_GPU_MANAGED amrex::Real FerroX::g11;
AMREX_GPU_MANAGED amrex::Real FerroX::g44;
AMREX_GPU_MANAGED amrex::Real FerroX::g44_p;
AMREX_GPU_MANAGED amrex::Real FerroX::g12;
AMREX_GPU_MANAGED amrex::Real FerroX::alpha_12;
AMREX_GPU_MANAGED amrex::Real FerroX::alpha_112;
AMREX_GPU_MANAGED amrex::Real FerroX::alpha_123;

// Constants for SC layer calculations
AMREX_GPU_MANAGED amrex::Real FerroX::Nc;
AMREX_GPU_MANAGED amrex::Real FerroX::Nv;
AMREX_GPU_MANAGED amrex::Real FerroX::Ec;
AMREX_GPU_MANAGED amrex::Real FerroX::Ev;
AMREX_GPU_MANAGED amrex::Real FerroX::q;
AMREX_GPU_MANAGED amrex::Real FerroX::kb;
AMREX_GPU_MANAGED amrex::Real FerroX::T;

// P and Phi Bc
AMREX_GPU_MANAGED int FerroX::P_BC_flag_lo;
AMREX_GPU_MANAGED int FerroX::P_BC_flag_hi;
AMREX_GPU_MANAGED amrex::Real FerroX::lambda;
AMREX_GPU_MANAGED amrex::Real FerroX::Phi_Bc_lo;
AMREX_GPU_MANAGED amrex::Real FerroX::Phi_Bc_hi;
AMREX_GPU_MANAGED amrex::Real FerroX::Phi_Bc_inc;
AMREX_GPU_MANAGED int FerroX::inc_step;

//problem type : initialization of P for 2D/3D/convergence problems
AMREX_GPU_MANAGED int FerroX::prob_type;

AMREX_GPU_MANAGED int FerroX::mlmg_verbosity;

AMREX_GPU_MANAGED int FerroX::TimeIntegratorOrder;

AMREX_GPU_MANAGED amrex::Real FerroX::delta;

void InitializeFerroXNamespace() {

     // ParmParse is way of reading inputs from the inputs file
     // pp.get means we require the inputs file to have it
     // pp.query means we optionally need the inputs file to have it - but we must supply a default here
     ParmParse pp;

     // We need to get n_cell from the inputs file - this is the number of cells on each side of
     amrex::Vector<int> temp_int(AMREX_SPACEDIM);
     if (pp.queryarr("n_cell",temp_int)) {
         for (int i=0; i<AMREX_SPACEDIM; ++i) {
             n_cell[i] = temp_int[i];
         }
     }

     // The domain is broken into boxes of size max_grid_size
     pp.get("max_grid_size",max_grid_size);

     pp.get("P_BC_flag_hi",P_BC_flag_hi); // 0 : P = 0, 1 : dp/dz = p/lambda, 2 : dp/dz = 0
     pp.get("P_BC_flag_lo",P_BC_flag_lo); // 0 : P = 0, 1 : dp/dz = p/lambda, 2 : dp/dz = 0
     pp.get("Phi_Bc_hi",Phi_Bc_hi);
     pp.get("Phi_Bc_lo",Phi_Bc_lo);

     Phi_Bc_inc = 0.;
     pp.query("Phi_Bc_inc",Phi_Bc_inc);

     inc_step = -1;
     pp.query("inc_step",inc_step);

     pp.get("TimeIntegratorOrder",TimeIntegratorOrder);

     pp.get("prob_type", prob_type);

     mlmg_verbosity = 1;
     pp.query("mlmg_verbosity",mlmg_verbosity);

     // Material Properties

     pp.get("epsilon_0",epsilon_0); // epsilon_0
     pp.get("epsilonX_fe",epsilonX_fe);// epsilon_r for FE
     pp.get("epsilonZ_fe",epsilonZ_fe);// epsilon_r for FE
     pp.get("epsilon_de",epsilon_de);// epsilon_r for DE
     pp.get("epsilon_si",epsilon_si);// epsilon_r for SC
     pp.get("alpha",alpha);
     pp.get("beta",beta);
     pp.get("gamma",FerroX::gamma);
     pp.get("alpha_12",alpha_12);
     pp.get("alpha_112",alpha_112);
     pp.get("alpha_123",alpha_123);
     pp.get("BigGamma",BigGamma);
     pp.get("g11",g11);
     pp.get("g44",g44);
     pp.get("g12",g12);
     pp.get("g44_p",g44_p);

     //stack thickness is assumed to be along z

     pp.get("DE_lo",DE_lo);
     pp.get("DE_hi",DE_hi);
     pp.get("FE_lo",FE_lo);
     pp.get("FE_hi",FE_hi);
     pp.get("SC_lo",SC_lo);
     pp.get("SC_hi",SC_hi);

     pp.get("lambda",lambda);

     // Default nsteps to 10, allow us to set it to something else in the inputs file
     nsteps = 10;
     pp.query("nsteps",nsteps);

     // Default plot_int to -1, allow us to set it to something else in the inputs file
     //  If plot_int < 0 then no plot files will be written
     plot_int = -1;
     pp.query("plot_int",plot_int);

     // time step
     pp.get("dt",dt);

     delta = 1.e-6;
     pp.query("delta",delta);

     amrex::Vector<amrex::Real> temp(AMREX_SPACEDIM);
     if (pp.queryarr("prob_lo",temp)) {
         for (int i=0; i<AMREX_SPACEDIM; ++i) {
             prob_lo[i] = temp[i];
         }
     }
     if (pp.queryarr("prob_hi",temp)) {
         for (int i=0; i<AMREX_SPACEDIM; ++i) {
             prob_hi[i] = temp[i];
         }
     }

     // For Silicon:
     // Nc = 2.8e25 m^-3
     // Nv = 1.04e25 m^-3
     // Band gap Eg = 1.12eV
     // 1eV = 1.602e-19 J

     Nc = 2.8e25;
     Nv = 1.04e25;
     Ec = 0.56;
     Ev = -0.56;
     q = 1.602e-19;
     kb = 1.38e-23; // Boltzmann constant
     T = 300; // Room Temp

}
