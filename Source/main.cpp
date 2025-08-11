#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLABecLaplacian.H>

#ifdef AMREX_USE_EB
#include <AMReX_MLEBABecLap.H>
#endif

#include <AMReX_MLMG.H> 
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>
#include "FerroX.H"
#include "Solver/ElectrostaticSolver.H"
#include "Solver/Initialization.H"
#include "Solver/ChargeDensity.H"
#include "Solver/TotalEnergyDensity.H"
#include "Input/BoundaryConditions/BoundaryConditions.H"
#include "Input/GeometryProperties/GeometryProperties.H"
#include "Utils/SelectWarpXUtils/WarpXUtil.H"
#include "Utils/SelectWarpXUtils/WarpXProfilerWrapper.H"
#include "Utils/eXstaticUtils/eXstaticUtil.H"
#include "Utils/FerroXUtils/FerroXUtil.H"




using namespace amrex;

using namespace FerroX;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);
    
    {
	    c_FerroX pFerroX;
            pFerroX.InitData();
            main_main(pFerroX);
    }
    amrex::Finalize();
    return 0;
}

void main_main (c_FerroX& rFerroX)
{

    BL_PROFILE("main()");

    Real total_step_strt_time = ParallelDescriptor::second();

    auto& rGprop = rFerroX.get_GeometryProperties();
    auto& geom = rGprop.geom;
    auto& ba = rGprop.ba;
    auto& dm = rGprop.dm;
    auto& is_periodic = rGprop.is_periodic;
    auto& prob_lo = rGprop.prob_lo;
    auto& prob_hi = rGprop.prob_hi;
    auto& n_cell = rGprop.n_cell;

    // read in inputs file
    InitializeFerroXNamespace(prob_lo, prob_hi);

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    MultiFab BigGamma(ba, dm, Ncomp, Nghost);
    MultiFab alpha(ba, dm, Ncomp, Nghost);
    MultiFab beta(ba, dm, Ncomp, Nghost);
    MultiFab gamma(ba, dm, Ncomp, Nghost);
    MultiFab epsilonX_fe(ba, dm, Ncomp, Nghost);
    MultiFab epsilonZ_fe(ba, dm, Ncomp, Nghost);
    MultiFab epsilon_de(ba, dm, Ncomp, Nghost);
    MultiFab epsilon_si(ba, dm, Ncomp, Nghost);
    MultiFab g11(ba, dm, Ncomp, Nghost);
    MultiFab g44(ba, dm, Ncomp, Nghost);
    MultiFab g44_p(ba, dm, Ncomp, Nghost);
    MultiFab g12(ba, dm, Ncomp, Nghost);
    MultiFab alpha_12(ba, dm, Ncomp, Nghost);
    MultiFab alpha_112(ba, dm, Ncomp, Nghost);
    MultiFab alpha_123(ba, dm, Ncomp, Nghost);

    Array<MultiFab, AMREX_SPACEDIM> P_old;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        P_old[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> P_new;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        P_new[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> P_new_pre;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        P_new_pre[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> GL_rhs;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        GL_rhs[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> GL_rhs_pre;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        GL_rhs_pre[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> GL_rhs_avg;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        GL_rhs_avg[dir].define(ba, dm, Ncomp, Nghost);
    }

    Array<MultiFab, AMREX_SPACEDIM> E;
    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        E[dir].define(ba, dm, Ncomp, 0);
    }

    MultiFab PoissonRHS(ba, dm, 1, 0);
    MultiFab PoissonPhi(ba, dm, 1, 1);
    MultiFab PoissonPhi_Old(ba, dm, 1, 1);
    MultiFab PoissonPhi_Prev(ba, dm, 1, 1);
    MultiFab PhiErr(ba, dm, 1, 1);
    MultiFab Phidiff(ba, dm, 1, 1);
    MultiFab Ex(ba, dm, 1, 0);
    MultiFab Ey(ba, dm, 1, 0);
    MultiFab Ez(ba, dm, 1, 0);

    MultiFab hole_den(ba, dm, 1, 0);
    MultiFab e_den(ba, dm, 1, 0);
    MultiFab charge_den(ba, dm, 1, 0);
    MultiFab MaterialMask(ba, dm, 1, 1);
    MultiFab tphaseMask(ba, dm, 1, 1);
    MultiFab angle_alpha(ba, dm, 1, 0);
    MultiFab angle_beta(ba, dm, 1, 0);
    MultiFab angle_theta(ba, dm, 1, 0);

    for (int dir = 0; dir < AMREX_SPACEDIM; dir++)
    {
        P_old[dir].setVal(0.);
        P_new[dir].setVal(0.);
        P_new_pre[dir].setVal(0.);
        GL_rhs[dir].setVal(0.);
        GL_rhs_pre[dir].setVal(0.);
        GL_rhs_avg[dir].setVal(0.);
        E[dir].setVal(0.);
    }

    e_den.setVal(0.);
    hole_den.setVal(0.);
    PoissonPhi.setVal(0.);
    PoissonRHS.setVal(0.);
    tphaseMask.setVal(0.);
    MaterialMask.setVal(0.);
    BigGamma.setVal(0.);
    alpha.setVal(0.);
    beta.setVal(0.);
    gamma.setVal(0.);
    epsilonX_fe.setVal(0.);
    epsilonZ_fe.setVal(0.);
    epsilon_de.setVal(0.);
    epsilon_si.setVal(0.);
    g11.setVal(0.);
    g44.setVal(0.);
    g44_p.setVal(0.);
    g12.setVal(0.);
    alpha_12.setVal(0.);
    alpha_112.setVal(0.);
    alpha_123.setVal(0.);
    angle_alpha.setVal(0.);
    angle_beta.setVal(0.);
    angle_theta.setVal(0.);

    //Initialize material mask
    InitializeMaterialMask(MaterialMask, geom, prob_lo, prob_hi);
    // Initialize material properties
    Initialize_MaterialProperties(rFerroX, geom, 
                                   BigGamma,
                                   alpha, 
                                   beta, 
                                   gamma, 
                                   epsilonX_fe, 
                                   epsilonZ_fe, 
                                   epsilon_de, 
                                   epsilon_si, 
                                   g11, 
                                   g44, 
                                   g44_p, 
                                   g12, 
                                   alpha_12, 
                                   alpha_112, 
                                   alpha_123);

    //check hardswitch_ratio and nucleation_ratio
    if (hardswitch_ratio > 1.0 || nucleation_ratio > 1.0 || (hardswitch_ratio + nucleation_ratio) > 1.0) {
        amrex::Abort("ERROR: hardswitch_ratio, nucleation_ratio, or their sum exceeds 1.0. Please check input parameters.");
    }
    
    // define hard to switch spots with larger alpha value
    if(hardswitch_flag == 1){
        printf("Set HardSwitch with ratio of:%g, with hardswitch_alpha_ratio:%g \n", hardswitch_ratio, hardswitch_alpha_ratio);
        SetHardToSwitchNucleation(alpha, MaterialMask, n_cell, hardswitch_ratio, hardswitch_alpha_ratio);
    }

    //InitializeMaterialMask(rFerroX, geom, MaterialMask);
    if(Coordinate_Transformation == 1){
       Initialize_tphase_Mask(rFerroX, geom, tphaseMask);
       Initialize_Euler_angles(rFerroX, geom, angle_alpha, angle_beta, angle_theta);
    }

    bool contains_SC = false;

    FerroX_Util::Contains_sc(MaterialMask, contains_SC);
    amrex::Print() << "contains_SC = " << contains_SC << "\n";

    std::array<std::array<amrex::LinOpBCType,AMREX_SPACEDIM>,2> LinOpBCType_2d;
    bool all_homogeneous_boundaries = true;
    bool some_functionbased_inhomogeneous_boundaries = false;
    bool some_constant_inhomogeneous_boundaries = false;

    SetPoissonBC(rFerroX, LinOpBCType_2d, all_homogeneous_boundaries, some_functionbased_inhomogeneous_boundaries, some_constant_inhomogeneous_boundaries);

    // coefficients for solver
    MultiFab alpha_cc(ba, dm, 1, 0);
    MultiFab beta_cc(ba, dm, 1, 1);
    std::array< MultiFab, AMREX_SPACEDIM > beta_face;
    AMREX_D_TERM(beta_face[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, 0);,
                 beta_face[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, 0);,
                 beta_face[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, 0););

    // set cell-centered beta coefficient to permittivity based on mask
    InitializePermittivity(LinOpBCType_2d, beta_cc, epsilonX_fe, epsilon_de, epsilon_si, MaterialMask, tphaseMask, n_cell, geom, prob_lo, prob_hi);
    eXstatic_MFab_Util::AverageCellCenteredMultiFabToCellFaces(beta_cc, beta_face);
    
    // time = starting time in the simulation
    Real time = 0.0;

    amrex::LPInfo info;
    std::unique_ptr<amrex::MLMG> pMLMG;
    std::unique_ptr<amrex::MLABecLaplacian> p_mlabec;
    int linop_maxorder = 2;
    int amrlev = 0; //refers to the setcoarsest level of the solve

    SetupMLMG(pMLMG, p_mlabec, LinOpBCType_2d, n_cell, beta_face, P_old, MaterialMask, rFerroX, PoissonPhi, time, info);

#ifdef AMREX_USE_EB
    std::unique_ptr<amrex::MLEBABecLap> p_mlebabec;
    SetupMLMG_EB(pMLMG, p_mlebabec, LinOpBCType_2d, n_cell, beta_face, beta_cc, rFerroX, PoissonPhi, time, info);
#endif
    
    // INITIALIZE P in FE and rho in SC regions

    //InitializePandRho(P_old, Gamma, charge_den, e_den, hole_den, geom, prob_lo, prob_hi);//old
    InitializePandRho(P_old, BigGamma, charge_den, e_den, hole_den, MaterialMask, tphaseMask, n_cell, geom, prob_lo, prob_hi);//mask based
    if(nucleation_flag == 1){
        printf("Set Nucleation with ratio of:%g, with hardswitch_ratio:%g \n", nucleation_ratio, hardswitch_ratio);
        SetNucleation(P_old, MaterialMask, n_cell, hardswitch_ratio, nucleation_ratio);
    }

#ifdef AMREX_USE_EB
    ComputePhi_Rho_EB(pMLMG, p_mlebabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr, 
                   P_old, charge_den, e_den, hole_den, MaterialMask, 
                   angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);
#else
    ComputePhi_Rho(pMLMG, p_mlabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr, 
                   P_old, charge_den, e_den, hole_den, MaterialMask, 
                   angle_alpha, angle_beta, angle_theta, geom, n_cell, prob_lo, prob_hi);
#endif

    // Calculate E from Phi
    ComputeEfromPhi(PoissonPhi, E, angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int plt_step = 0;
        WritePlotfile(rFerroX, PoissonPhi, PoissonRHS, P_old, E, hole_den, e_den, charge_den, beta_cc, 
                      MaterialMask, tphaseMask,
                      BigGamma,
                      alpha, 
                      beta, 
                      gamma, 
                      epsilonX_fe, 
                      epsilonZ_fe, 
                      epsilon_de, 
                      epsilon_si, 
                      g11, 
                      g44, 
                      g44_p, 
                      g12, 
                      alpha_12, 
                      alpha_112, 
                      alpha_123, angle_alpha, angle_beta, angle_theta, Phidiff, geom, time, plt_step);
    }

    amrex::Print() << "\n ========= Advance Steps  ========== \n"<< std::endl;

    int steady_state_step = 1000000; //Initialize to a large number. It will be overwritten by the time step at which steady state condition is satidfied

    int sign = 1; //change sign to -1*sign whenever abs(Phi_Bc_hi) == Phi_Bc_hi_max to do triangular wave sweep
    int num_Vapp = 0;
    Real tiny = 1.e-6;    
 
    for (int step = 1; step <= nsteps; ++step)
    {
        Real step_strt_time = ParallelDescriptor::second();

        // compute f^n = f(P^n,Phi^n)
        CalculateTDGL_RHS(GL_rhs, P_old, E, BigGamma,  alpha, beta, gamma, g11, g44, g44_p, g12, alpha_12, alpha_112, alpha_123, MaterialMask, tphaseMask, angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);

        // P^{n+1,*} = P^n + dt * f^n
        for (int i = 0; i < 3; i++){
            MultiFab::LinComb(P_new_pre[i], 1.0, P_old[i], 0, dt, GL_rhs[i], 0, 0, 1, Nghost);
            P_new_pre[i].FillBoundary(geom.periodicity()); 
        }  
	

#ifdef AMREX_USE_EB
        ComputePhi_Rho_EB(pMLMG, p_mlebabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr, 
                   P_new_pre, charge_den, e_den, hole_den, MaterialMask, 
                   angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);
#else
        ComputePhi_Rho(pMLMG, p_mlabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr, 
                   P_new_pre, charge_den, e_den, hole_den, MaterialMask, 
                   angle_alpha, angle_beta, angle_theta, geom,  n_cell, prob_lo, prob_hi);
#endif
        
        if (TimeIntegratorOrder == 1) {

            // copy new solution into old solution
            for (int i = 0; i < 3; i++){
                MultiFab::Copy(P_old[i], P_new_pre[i], 0, 0, 1, 0);
                // fill periodic ghost cells
                P_old[i].FillBoundary(geom.periodicity());
                P_new_pre[i].FillBoundary(geom.periodicity());
            }
            if(nucleation_flag == 1){
                printf("Set Nucleation with ratio of:%g, with hardswitch_ratio:%g \n", nucleation_ratio, hardswitch_ratio);
                SetNucleation(P_old, MaterialMask, n_cell, hardswitch_ratio, nucleation_ratio);
            }
            
        } else {
        
            // compute f^{n+1,*} = f(P^{n+1,*},Phi^{n+1,*})
            CalculateTDGL_RHS(GL_rhs_pre, P_new_pre, E, BigGamma,  alpha, beta, gamma, g11, g44, g44_p, g12, alpha_12, alpha_112, alpha_123, MaterialMask, tphaseMask, angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);

            // P^{n+1} = P^n + dt/2 * f^n + dt/2 * f^{n+1,*}
            for (int i = 0; i < 3; i++){
                MultiFab::LinComb(GL_rhs_avg[i], 0.5, GL_rhs[i], 0, 0.5, GL_rhs_pre[i], 0, 0, 1, Nghost);    
                MultiFab::LinComb(P_new[i], 1.0, P_old[i], 0, dt, GL_rhs_avg[i], 0, 0, 1, Nghost);
            }
        
#ifdef AMREX_USE_EB
            ComputePhi_Rho_EB(pMLMG, p_mlebabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr, 
                   P_new, charge_den, e_den, hole_den, MaterialMask, 
                   angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);
#else
            ComputePhi_Rho(pMLMG, p_mlabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr, 
                   P_new, charge_den, e_den, hole_den, MaterialMask, 
                   angle_alpha, angle_beta, angle_theta, geom,  n_cell, prob_lo, prob_hi);
#endif

            // copy new solution into old solution
            for (int i = 0; i < 3; i++){
                MultiFab::Copy(P_old[i], P_new[i], 0, 0, 1, 0);
                // fill periodic ghost cells
                P_old[i].FillBoundary(geom.periodicity());
            }
    	}

        // Check if steady state has reached 
        CheckSteadyState(PoissonPhi, PoissonPhi_Old, Phidiff, phi_tolerance, step, steady_state_step, inc_step);

	    // Calculate E from Phi
	    ComputeEfromPhi(PoissonPhi, E, angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);


	    Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";
        amrex::Print() << " \n";

        // update time
        time = time + dt;


        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && (step%plot_int == 0 || step == steady_state_step))
        {
            int plt_step = step;
            WritePlotfile(rFerroX, PoissonPhi, PoissonRHS, P_old, E, hole_den, e_den, charge_den, beta_cc, 
                      MaterialMask, tphaseMask,
                      BigGamma,
                      alpha, 
                      beta, 
                      gamma, 
                      epsilonX_fe, 
                      epsilonZ_fe, 
                      epsilon_de, 
                      epsilon_si, 
                      g11, 
                      g44, 
                      g44_p, 
                      g12, 
                      alpha_12, 
                      alpha_112, 
                      alpha_123, angle_alpha, angle_beta, angle_theta, Phidiff, geom, time, plt_step);
            
        }

        if(voltage_sweep == 1 && inc_step > 0 && step == inc_step)
        {
           //Update time-dependent Boundary Condition of Poisson's equation

            Phi_Bc_hi += sign*Phi_Bc_inc;
            num_Vapp += 1;
            if(std::abs(std::abs(Phi_Bc_hi) - Phi_Bc_hi_max) <= tiny) {
              sign *= -1;
              amrex::Print() << "Direction of voltage sweep is reversed. Phi_Bc_hi = " << Phi_Bc_hi << ", and Phi_Bc_hi_max = " << Phi_Bc_hi_max << std::endl;
            }
            amrex::Print() << "step = " << step << ", Phi_Bc_hi = " << Phi_Bc_hi << ", num_Vapp = " << num_Vapp << ", sign = " << sign << std::endl;

            // Set Dirichlet BC for Phi in z
            //SetPhiBC_z(PoissonPhi, P_old, MaterialMask, n_cell, geom);
            SetPhiBC_z(PoissonPhi, n_cell, geom);

           // set Dirichlet BC by reading in the ghost cell values
#ifdef AMREX_USE_EB
           p_mlebabec->setLevelBC(amrlev, &PoissonPhi);
#else 
           p_mlabec->setLevelBC(amrlev, &PoissonPhi);
#endif

#ifdef AMREX_USE_EB
           ComputePhi_Rho_EB(pMLMG, p_mlebabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr, 
                   P_old, charge_den, e_den, hole_den, MaterialMask, 
                   angle_alpha, angle_beta, angle_theta, geom, prob_lo, prob_hi);
#else
           ComputePhi_Rho(pMLMG, p_mlabec, alpha_cc, PoissonRHS, PoissonPhi, PoissonPhi_Prev, PhiErr, 
                   P_old, charge_den, e_den, hole_den, MaterialMask, 
                   angle_alpha, angle_beta, angle_theta, geom,  n_cell, prob_lo, prob_hi);
#endif
           
        }//end inc_step	
   
        if (voltage_sweep == 0 && step == steady_state_step) {
           amrex::Print() << "voltage_sweep == 0 && step == steady_state_step!" << "\n";   
           break;
        }
        if (voltage_sweep == 1 && Phi_Bc_hi > 0. && Phi_Bc_hi - Phi_Bc_hi_max > tiny) {
           amrex::Print() << "voltage_sweep == 1 && Phi_Bc_hi > 0. && Phi_Bc_hi - Phi_Bc_hi_max > tiny!" << "\n";
           break;
        }        
        if (voltage_sweep == 1 && Phi_Bc_hi < 0. && -Phi_Bc_hi - Phi_Bc_hi_max > tiny) {
           amrex::Print() << "voltage_sweep == 1 && Phi_Bc_hi < 0. && -Phi_Bc_hi - Phi_Bc_hi_max > tiny!" << "\n";
           break;
        }   
        if (voltage_sweep == 1 && num_Vapp == num_Vapp_max) {
           amrex::Print() << "voltage_sweep == 1 && num_Vapp == num_Vapp_max!"  << "\n";
           break;
        }

    } // end step

    // MultiFab memory usage
    const int IOProc = ParallelDescriptor::IOProcessorNumber();

    amrex::Long min_fab_megabytes  = amrex::TotalBytesAllocatedInFabsHWM()/1048576;
    amrex::Long max_fab_megabytes  = min_fab_megabytes;

    ParallelDescriptor::ReduceLongMin(min_fab_megabytes, IOProc);
    ParallelDescriptor::ReduceLongMax(max_fab_megabytes, IOProc);

    amrex::Print() << "High-water FAB megabyte spread across MPI nodes: ["
                   << min_fab_megabytes << " ... " << max_fab_megabytes << "]\n";

    min_fab_megabytes  = amrex::TotalBytesAllocatedInFabs()/1048576;
    max_fab_megabytes  = min_fab_megabytes;

    ParallelDescriptor::ReduceLongMin(min_fab_megabytes, IOProc);
    ParallelDescriptor::ReduceLongMax(max_fab_megabytes, IOProc);

    amrex::Print() << "Curent     FAB megabyte spread across MPI nodes: ["
                   << min_fab_megabytes << " ... " << max_fab_megabytes << "]\n";
    
    Real total_step_stop_time = ParallelDescriptor::second() - total_step_strt_time;
    ParallelDescriptor::ReduceRealMax(total_step_stop_time);

    amrex::Print() << "Total run time " << total_step_stop_time << " seconds\n";

}
