namespace FerroX {

    extern AMREX_GPU_MANAGED int max_grid_size;
    extern AMREX_GPU_MANAGED int nsteps;
    extern AMREX_GPU_MANAGED int plot_int;

    // time step
    extern AMREX_GPU_MANAGED amrex::Real dt;

    extern AMREX_GPU_MANAGED amrex::GpuArray<int, AMREX_SPACEDIM> n_cell; // number of cells in each direction
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo; // physical lo coordinate
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi; // physical hi coordinate

    // multimaterial stack geometry
    extern AMREX_GPU_MANAGED amrex::Real DE_lo;
    extern AMREX_GPU_MANAGED amrex::Real FE_lo;
    extern AMREX_GPU_MANAGED amrex::Real SC_lo;
    extern AMREX_GPU_MANAGED amrex::Real DE_hi;
    extern AMREX_GPU_MANAGED amrex::Real FE_hi;
    extern AMREX_GPU_MANAGED amrex::Real SC_hi;

    // material parameters
    extern AMREX_GPU_MANAGED amrex::Real epsilon_0;
    extern AMREX_GPU_MANAGED amrex::Real epsilonX_fe;
    extern AMREX_GPU_MANAGED amrex::Real epsilonZ_fe;
    extern AMREX_GPU_MANAGED amrex::Real epsilon_de;
    extern AMREX_GPU_MANAGED amrex::Real epsilon_si;
    extern AMREX_GPU_MANAGED amrex::Real alpha; // alpha = 2*alpha_1
    extern AMREX_GPU_MANAGED amrex::Real beta; // beta = 4*alpha_11
    extern AMREX_GPU_MANAGED amrex::Real gamma; // gamma = 6*alpha_111
    extern AMREX_GPU_MANAGED amrex::Real BigGamma;
    extern AMREX_GPU_MANAGED amrex::Real g11;
    extern AMREX_GPU_MANAGED amrex::Real g44;
    extern AMREX_GPU_MANAGED amrex::Real g44_p;
    extern AMREX_GPU_MANAGED amrex::Real g12;
    extern AMREX_GPU_MANAGED amrex::Real alpha_12;
    extern AMREX_GPU_MANAGED amrex::Real alpha_112;
    extern AMREX_GPU_MANAGED amrex::Real alpha_123;

    // Constants for SC layer calculations
    extern AMREX_GPU_MANAGED amrex::Real Nc;
    extern AMREX_GPU_MANAGED amrex::Real Nv;
    extern AMREX_GPU_MANAGED amrex::Real Ec;
    extern AMREX_GPU_MANAGED amrex::Real Ev;
    extern AMREX_GPU_MANAGED amrex::Real q;
    extern AMREX_GPU_MANAGED amrex::Real kb;
    extern AMREX_GPU_MANAGED amrex::Real T;

    // P and Phi Bc
    extern AMREX_GPU_MANAGED int P_BC_flag_lo;
    extern AMREX_GPU_MANAGED int P_BC_flag_hi;
    extern AMREX_GPU_MANAGED amrex::Real lambda;
    extern AMREX_GPU_MANAGED amrex::Real Phi_Bc_lo;
    extern AMREX_GPU_MANAGED amrex::Real Phi_Bc_hi;
    extern AMREX_GPU_MANAGED amrex::Real Phi_Bc_inc;
    extern AMREX_GPU_MANAGED int inc_step;
    extern AMREX_GPU_MANAGED int inc_step_sign_change;

    //problem type : initialization of P for 2D/3D/convergence problems
    extern AMREX_GPU_MANAGED int prob_type;

    extern AMREX_GPU_MANAGED int mlmg_verbosity;

    extern AMREX_GPU_MANAGED int TimeIntegratorOrder;

    extern AMREX_GPU_MANAGED amrex::Real delta;
}