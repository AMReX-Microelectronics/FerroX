#include "FerroX.H"

#include <AMReX_ParmParse.H>

int FerroX::max_grid_size = 0;
int FerroX::nsteps = 0 ;
int FerroX::plot_int = 0;

// time step
amrex::Real FerroX::dt = 0.;

amrex::Vector<int> FerroX::bc_lo(AMREX_SPACEDIM,0);
amrex::Vector<int> FerroX::bc_hi(AMREX_SPACEDIM,0);

amrex::GpuArray<int, AMREX_SPACEDIM> FerroX::n_cell; // number of cells in each direction
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> FerroX::prob_lo; // physical lo coordinate
amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> FerroX::prob_hi; // physical hi coordinate

// multimaterial stack geometry
amrex::Real FerroX::FerroX::DE_lo = 0.;
amrex::Real FerroX::FE_lo = 0.;
amrex::Real FerroX::FerroX::SC_lo = 0.;
amrex::Real FerroX::DE_hi = 0.;
amrex::Real FerroX::FE_hi = 0.;
amrex::Real FerroX::SC_hi = 0.;

// material parameters
amrex::Real FerroX::epsilon_0 = 0.;
amrex::Real FerroX::epsilonX_fe = 0.;
amrex::Real FerroX::epsilonZ_fe = 0.;
amrex::Real FerroX::epsilon_de = 0.;
amrex::Real FerroX::epsilon_si = 0.;
amrex::Real FerroX::alpha = 0.; // alpha = 2*alpha_1
amrex::Real FerroX::beta = 0.; // beta = 4*alpha_11
amrex::Real FerroX::gamma = 0.; // gamma = 6*alpha_111
amrex::Real FerroX::BigGamma = 0.;
amrex::Real FerroX::g11 = 0.;
amrex::Real FerroX::g44 = 0.;
amrex::Real FerroX::g44_p = 0.;
amrex::Real FerroX::g12 = 0.;
amrex::Real FerroX::alpha_12 = 0.;
amrex::Real FerroX::alpha_112 = 0.;
amrex::Real FerroX::alpha_123 = 0.;

// Constants for SC layer calculations
amrex::Real FerroX::Nc = 0.;
amrex::Real FerroX::Nv = 0.;
amrex::Real FerroX::Ec = 0.;
amrex::Real FerroX::Ev = 0.;
amrex::Real FerroX::q = 0.;
amrex::Real FerroX::kb = 0.;
amrex::Real FerroX::T = 0.;

// P and Phi Bc
int FerroX::P_BC_flag_lo = 0;
int FerroX::P_BC_flag_hi = 0;
amrex::Real FerroX::lambda = 0.;
amrex::Real FerroX::Phi_Bc_lo = 0.;
amrex::Real FerroX::Phi_Bc_hi = 0.;
amrex::Real FerroX::Phi_Bc_inc = 0.;
int FerroX::inc_step = 0;

//problem type : initialization of P for 2D/3D/convergence problems
int FerroX::prob_type = 0;

int FerroX::mlmg_verbosity = 0;

int FerroX::TimeIntegratorOrder = 0;

amrex::Real FerroX::delta = 0.;

void InitializeGlobalVariables() {

     // ParmParse is way of reading inputs from the inputs file
     // pp.get means we require the inputs file to have it
     // pp.query means we optionally need the inputs file to have it - but we must supply a default here
     ParmParse pp;

     // We need to get n_cell from the inputs file - this is the number of cells on each side of
     amrex::Vector<int> temp_int(AMREX_SPACEDIM);
     if (pp.queryarr("n_cell",temp_int)) {
         for (int i=0; i<AMREX_SPACEDIM; ++i) {
             FerroX::n_cell[i] = temp_int[i];
         }
     }

     // The domain is broken into boxes of size max_grid_size
     pp.get("max_grid_size",FerroX::max_grid_size);

     pp.get("P_BC_flag_hi",FerroX::P_BC_flag_hi); // 0 : P = 0, 1 : dp/dz = p/lambda, 2 : dp/dz = 0
     pp.get("P_BC_flag_lo",FerroX::P_BC_flag_lo); // 0 : P = 0, 1 : dp/dz = p/lambda, 2 : dp/dz = 0
     pp.get("Phi_Bc_hi",FerroX::Phi_Bc_hi);
     pp.get("Phi_Bc_lo",FerroX::Phi_Bc_lo);

     FerroX::Phi_Bc_inc = 0.;
     pp.query("Phi_Bc_inc",FerroX::Phi_Bc_inc);

     FerroX::inc_step = -1;
     pp.query("inc_step",FerroX::inc_step);

     pp.get("TimeIntegratorOrder",FerroX::TimeIntegratorOrder);

     pp.get("prob_type", FerroX::prob_type);

     FerroX::mlmg_verbosity = 1;
     pp.query("mlmg_verbosity",FerroX::mlmg_verbosity);

     // Material Properties

     pp.get("epsilon_0",FerroX::epsilon_0); // epsilon_0
     pp.get("epsilonX_fe",FerroX::epsilonX_fe);// epsilon_r for FE
     pp.get("epsilonZ_fe",FerroX::epsilonZ_fe);// epsilon_r for FE
     pp.get("epsilon_de",FerroX::epsilon_de);// epsilon_r for DE
     pp.get("epsilon_si",FerroX::epsilon_si);// epsilon_r for SC
     pp.get("alpha",FerroX::alpha);
     pp.get("beta",FerroX::beta);
     pp.get("gamma",FerroX::gamma);
     pp.get("alpha_12",FerroX::alpha_12);
     pp.get("alpha_112",FerroX::alpha_112);
     pp.get("alpha_123",FerroX::alpha_123);
     pp.get("BigGamma",FerroX::BigGamma);
     pp.get("g11",FerroX::g11);
     pp.get("g44",FerroX::g44);
     pp.get("g12",FerroX::g12);
     pp.get("g44_p",FerroX::g44_p);

     //stack thickness is assumed to be along z

     pp.get("DE_lo",FerroX::DE_lo);
     pp.get("DE_hi",FerroX::DE_hi);
     pp.get("FE_lo",FerroX::FE_lo);
     pp.get("FE_hi",FerroX::FE_hi);
     pp.get("SC_lo",FerroX::SC_lo);
     pp.get("SC_hi",FerroX::SC_hi);

     pp.get("lambda",FerroX::lambda);

     // Default nsteps to 10, allow us to set it to something else in the inputs file
     FerroX::nsteps = 10;
     pp.query("nsteps",FerroX::nsteps);

     // Default plot_int to -1, allow us to set it to something else in the inputs file
     //  If plot_int < 0 then no plot files will be written
     FerroX::plot_int = -1;
     pp.query("plot_int",FerroX::plot_int);

     // time step
     pp.get("dt",FerroX::dt);

     FerroX::delta = 1.e-6;
     pp.query("delta",FerroX::delta);

     amrex::Vector<amrex::Real> temp(AMREX_SPACEDIM);
     if (pp.queryarr("prob_lo",temp)) {
         for (int i=0; i<AMREX_SPACEDIM; ++i) {
             FerroX::prob_lo[i] = temp[i];
         }
     }
     if (pp.queryarr("prob_hi",temp)) {
         for (int i=0; i<AMREX_SPACEDIM; ++i) {
             FerroX::prob_hi[i] = temp[i];
         }
     }

     // For Silicon:
     // Nc = 2.8e25 m^-3
     // Nv = 1.04e25 m^-3
     // Band gap Eg = 1.12eV
     // 1eV = 1.602e-19 J

     FerroX::Nc = 2.8e25;
     FerroX::Nv = 1.04e25;
     FerroX::Ec = 0.56;
     FerroX::Ev = -0.56;
     FerroX::q = 1.602e-19;
     FerroX::kb = 1.38e-23; // Boltzmann constant
     FerroX::T = 300; // Room Temp

}
