namespace FerroX {
    
    extern AMREX_GPU_MANAGED int nsteps;
    extern AMREX_GPU_MANAGED int plot_int;

    extern AMREX_GPU_MANAGED int hardswitch_flag;
    extern AMREX_GPU_MANAGED int nucleation_flag;

    
    extern AMREX_GPU_MANAGED amrex::Real hardswitch_ratio;
    extern AMREX_GPU_MANAGED amrex::Real nucleation_ratio;
    extern AMREX_GPU_MANAGED amrex::Real hardswitch_alpha_ratio;


    extern int plot_Phi;
    extern int plot_PoissonRHS;
    extern int plot_E;
    extern int plot_holes;
    extern int plot_electrons;
    extern int plot_charge;
    extern int plot_epsilon;
    extern int plot_mask;
    extern int plot_tphase;
    extern int plot_mat_BigGamma;
    extern int plot_mat_alpha;
    extern int plot_mat_beta;
    extern int plot_mat_gamma;
    extern int plot_mat_epsilonX_fe;
    extern int plot_mat_epsilonZ_fe;
    extern int plot_mat_epsilon_de;
    extern int plot_mat_epsilon_si;
    extern int plot_mat_g11;
    extern int plot_mat_g44;
    extern int plot_mat_g44_p;
    extern int plot_mat_g12;
    extern int plot_mat_alpha_12;
    extern int plot_mat_alpha_112;
    extern int plot_mat_alpha_123;
    extern int plot_angle_alpha;
    extern int plot_angle_beta;
    extern int plot_angle_theta;
    extern int plot_PhiDiff;

    // time step
    extern AMREX_GPU_MANAGED amrex::Real dt;

    // multimaterial stack geometry
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> DE_lo;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> DE1_lo;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> FE_lo;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> SC_lo;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> DE_hi;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> DE1_hi;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> FE_hi;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> SC_hi;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> Channel_hi;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> Channel_lo;

    // // material parameters
    extern AMREX_GPU_MANAGED amrex::Real epsilon_0;
    // extern AMREX_GPU_MANAGED amrex::Real epsilonX_fe;
    extern AMREX_GPU_MANAGED amrex::Real epsilonX_fe_tphase;
    // extern AMREX_GPU_MANAGED amrex::Real epsilonZ_fe;
    // extern AMREX_GPU_MANAGED amrex::Real epsilon_de;
    // extern AMREX_GPU_MANAGED amrex::Real epsilon_si;
    // extern AMREX_GPU_MANAGED amrex::Real alpha; // alpha = 2*alpha_1
    // extern AMREX_GPU_MANAGED amrex::Real beta; // beta = 4*alpha_11
    // extern AMREX_GPU_MANAGED amrex::Real gamma; // gamma = 6*alpha_111
    // extern AMREX_GPU_MANAGED amrex::Real BigGamma;
    // extern AMREX_GPU_MANAGED amrex::Real g11;
    // extern AMREX_GPU_MANAGED amrex::Real g44;
    // extern AMREX_GPU_MANAGED amrex::Real g44_p;
    // extern AMREX_GPU_MANAGED amrex::Real g12;
    // extern AMREX_GPU_MANAGED amrex::Real alpha_12;
    // extern AMREX_GPU_MANAGED amrex::Real alpha_112;
    // extern AMREX_GPU_MANAGED amrex::Real alpha_123;

    // Constants for SC layer calculations
    extern AMREX_GPU_MANAGED amrex::Real Nc;
    extern AMREX_GPU_MANAGED amrex::Real Nv;
    extern AMREX_GPU_MANAGED amrex::Real bandgap;
    extern AMREX_GPU_MANAGED amrex::Real affinity;
    extern AMREX_GPU_MANAGED amrex::Real q;
    extern AMREX_GPU_MANAGED amrex::Real kb;
    extern AMREX_GPU_MANAGED amrex::Real T;
    extern AMREX_GPU_MANAGED amrex::Real acceptor_doping;
    extern AMREX_GPU_MANAGED amrex::Real donor_doping;
    extern AMREX_GPU_MANAGED amrex::Real acceptor_ionization_energy;
    extern AMREX_GPU_MANAGED amrex::Real donor_ionization_energy;
    extern AMREX_GPU_MANAGED amrex::Real intrinsic_carrier_concentration;
    extern AMREX_GPU_MANAGED int use_Fermi_Dirac;
    extern AMREX_GPU_MANAGED int use_work_function;
    extern AMREX_GPU_MANAGED amrex::Real metal_work_function;
    

    // P and Phi Bc
    extern AMREX_GPU_MANAGED amrex::Real lambda;

    extern AMREX_GPU_MANAGED amrex::GpuArray<int, AMREX_SPACEDIM> P_BC_flag_lo;
    extern AMREX_GPU_MANAGED amrex::GpuArray<int, AMREX_SPACEDIM> P_BC_flag_hi;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> Remnant_P;
    extern AMREX_GPU_MANAGED amrex::Real noise_amplitude;
    extern AMREX_GPU_MANAGED amrex::Real theta_dep;

    //problem type : initialization of P for 2D/3D/convergence problems
    extern AMREX_GPU_MANAGED int prob_type;
    extern AMREX_GPU_MANAGED int is_polarization_scalar;

    extern AMREX_GPU_MANAGED int mlmg_verbosity;

    extern AMREX_GPU_MANAGED int TimeIntegratorOrder;

    extern AMREX_GPU_MANAGED amrex::Real delta;

    extern AMREX_GPU_MANAGED int Coordinate_Transformation;
    extern AMREX_GPU_MANAGED int use_Euler_angles;
    
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> t_phase_lo;
    extern AMREX_GPU_MANAGED amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> t_phase_hi;

    //Params related to voltage sweep
    extern AMREX_GPU_MANAGED int voltage_sweep;
    extern AMREX_GPU_MANAGED int inc_step;
    extern AMREX_GPU_MANAGED amrex::Real Phi_Bc_lo;
    extern AMREX_GPU_MANAGED amrex::Real Phi_Bc_hi;
    extern AMREX_GPU_MANAGED amrex::Real Phi_Bc_inc;
    extern AMREX_GPU_MANAGED amrex::Real Phi_Bc_hi_max;
    extern AMREX_GPU_MANAGED amrex::Real phi_tolerance;
    extern AMREX_GPU_MANAGED int num_Vapp_max;

    extern AMREX_GPU_MANAGED int random_seed;
}
