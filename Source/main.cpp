
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

    Real total_step_strt_time = ParallelDescriptor::second();

    // read in inputs file
    InitializeFerroXNamespace();

    auto& rGprop = rFerroX.get_GeometryProperties();
    auto& geom = rGprop.geom;
    auto& ba = rGprop.ba;
    auto& dm = rGprop.dm;
    auto& is_periodic = rGprop.is_periodic;


    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    MultiFab Gamma(ba, dm, Ncomp, Nghost);

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

    MultiFab PoissonRHS(ba, dm, 1, 0);
    MultiFab PoissonPhi(ba, dm, 1, 1);
    MultiFab PoissonPhi_Prev(ba, dm, 1, 1);
    MultiFab PhiErr(ba, dm, 1, 1);
    MultiFab Ex(ba, dm, 1, 0);
    MultiFab Ey(ba, dm, 1, 0);
    MultiFab Ez(ba, dm, 1, 0);

    MultiFab hole_den(ba, dm, 1, 0);
    MultiFab e_den(ba, dm, 1, 0);
    MultiFab charge_den(ba, dm, 1, 0);


    //Solver for Poisson equation
    LPInfo info;
    std::unique_ptr<amrex::MLMG> pMLMG;
    // order of stencil
    int linop_maxorder = 2;
    std::array<std::array<amrex::LinOpBCType,AMREX_SPACEDIM>,2> LinOpBCType_2d;
    bool all_homogeneous_boundaries = true;
    bool some_functionbased_inhomogeneous_boundaries = false;
    bool some_constant_inhomogeneous_boundaries = false;

#ifdef AMREX_USE_EB
    MultiFab Plt(ba, dm, 14, 0,  MFInfo(), *rGprop.pEB->p_factory_union);
#else    
    MultiFab Plt(ba, dm, 15, 0);
#endif

    SetPoissonBC(rFerroX, LinOpBCType_2d, all_homogeneous_boundaries, some_functionbased_inhomogeneous_boundaries, some_constant_inhomogeneous_boundaries);

    // coefficients for solver
    MultiFab alpha_cc(ba, dm, 1, 0);
    MultiFab beta_cc(ba, dm, 1, 0);
    std::array< MultiFab, AMREX_SPACEDIM > beta_face;
    AMREX_D_TERM(beta_face[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, 0);,
                 beta_face[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, 0);,
                 beta_face[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, 0););

    // set face-centered beta coefficient to 
    // epsilon values in SC, FE, and DE layers
    InitializePermittivity(beta_face, geom);
    //beta_face[0].setVal(10.0);
    //beta_face[1].setVal(10.0);
    //beta_face[2].setVal(10.0);
    //InitializePermittivity(beta_cc, geom);
    //Multifab_Manipulation::AverageCellCenteredMultiFabToCellFaces(beta_cc, beta_face);
    int amrlev = 0; //refers to the setcoarsest level of the solve


#ifdef AMREX_USE_EB

    std::unique_ptr<amrex::MLEBABecLap> p_mlebabec;
    p_mlebabec = std::make_unique<amrex::MLEBABecLap>();
    p_mlebabec->define({geom}, {ba}, {dm}, info,{& *rGprop.pEB->p_factory_union});

    // Force singular system to be solvable
    p_mlebabec->setEnforceSingularSolvable(false);

    // set order of stencil
    p_mlebabec->setMaxOrder(linop_maxorder);

    // assign domain boundary conditions to the solver
    // see Src/Boundary/AMReX_LO_BCTYPES.H for supported types
    p_mlebabec->setDomainBC(LinOpBCType_2d[0], LinOpBCType_2d[1]);

    if(some_constant_inhomogeneous_boundaries)
    {
        Fill_Constant_Inhomogeneous_Boundaries(rFerroX, PoissonPhi);
    }
    if(some_functionbased_inhomogeneous_boundaries)
    {
        Fill_FunctionBased_Inhomogeneous_Boundaries(rFerroX, PoissonPhi);
        //Note that previously in c_BoundaryCondition constructor, it has been asserted
        //that the use of robin is not supported with embedded boundaries.
    }
    PoissonPhi.FillBoundary(geom.periodicity());

    // Set Dirichlet BC for Phi in z
    //SetPhiBC_z(PoissonPhi); 
    p_mlebabec->setLevelBC(amrlev, &PoissonPhi);
    
    // (A*alpha_cc - B * div beta grad) phi = rhs
    p_mlebabec->setScalars(-1.0, 1.0); // A = -1.0, B = 1.0; solving (-alpha - div beta grad) phi = RHS
    p_mlebabec->setBCoeffs(amrlev, amrex::GetArrOfConstPtrs(beta_face));

    // set alpha, and beta_fc coefficients
    //p_mlebabec->setACoeffs(amrlev, alpha_cc);

    //Multifab_Manipulation::AverageCellCenteredMultiFabToCellFaces(beta_cc, beta_face);
    Multifab_Manipulation::AverageFaceCenteredMultiFabToCellCenters(beta_face, beta_cc);
    if(rGprop.pEB->specify_inhomogeneous_dirichlet == 0)
    {
        //p_mlebabec->setEBHomogDirichlet(amrlev, *rGprop.pEB->p_surf_beta_union);
        p_mlebabec->setEBHomogDirichlet(amrlev, beta_cc);
    }
    else
    {
        //p_mlebabec->setEBDirichlet(amrlev, *rGprop.pEB->p_surf_soln_union, *rGprop.pEB->p_surf_beta_union);
        p_mlebabec->setEBDirichlet(amrlev, *rGprop.pEB->p_surf_soln_union, beta_cc);
    }

    pMLMG = std::make_unique<MLMG>(*p_mlebabec);

    pMLMG->setVerbose(mlmg_verbosity);
#else
    std::unique_ptr<amrex::MLABecLaplacian> p_mlabec;
    p_mlabec = std::make_unique<amrex::MLABecLaplacian>();
    p_mlabec->define({geom}, {ba}, {dm}, info);

    //Force singular system to be solvable
    p_mlabec->setEnforceSingularSolvable(false); 

    p_mlabec->setMaxOrder(linop_maxorder);  

    p_mlabec->setDomainBC(LinOpBCType_2d[0], LinOpBCType_2d[1]);

    if(some_constant_inhomogeneous_boundaries)
    {
        Fill_Constant_Inhomogeneous_Boundaries(rFerroX, PoissonPhi);
    }
    if(some_functionbased_inhomogeneous_boundaries)
    {
        Fill_FunctionBased_Inhomogeneous_Boundaries(rFerroX, PoissonPhi);
    }
    PoissonPhi.FillBoundary(geom.periodicity());

    // set Dirichlet BC by reading in the ghost cell values
    p_mlabec->setLevelBC(amrlev, &PoissonPhi);
    
    // (A*alpha_cc - B * div beta grad) phi = rhs
    p_mlabec->setScalars(-1.0, 1.0); // A = -1.0, B = 1.0; solving (-alpha - div beta grad) phi = RHS
    p_mlabec->setBCoeffs(amrlev, amrex::GetArrOfConstPtrs(beta_face));

    //Declare MLMG object
    pMLMG = std::make_unique<MLMG>(*p_mlabec);
    pMLMG->setVerbose(mlmg_verbosity);
#endif


    // time = starting time in the simulation
    Real time = 0.0;

    // INITIALIZE P in FE and rho in SC regions

    InitializePandRho(P_old, Gamma, charge_den, e_den, hole_den, geom);

    //Obtain self consisten Phi and rho
    Real tol = 1.e-5;
    Real err = 1.0;
    int iter = 0;
    
    //while(iter < 2){
    while(err > tol){
   
	//Compute RHS of Poisson equation
	ComputePoissonRHS(PoissonRHS, P_old, charge_den, geom);

        dF_dPhi(alpha_cc, PoissonRHS, PoissonPhi, P_old, charge_den, e_den, hole_den, geom);

        ComputePoissonRHS_Newton(PoissonRHS, PoissonPhi, alpha_cc); 

#ifdef AMREX_USE_EB
        p_mlebabec->setACoeffs(0, alpha_cc);
#else
        p_mlabec->setACoeffs(0, alpha_cc);
#endif
        //Initial guess for phi
        PoissonPhi.setVal(0.);

        //Poisson Solve
        pMLMG->solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
	PoissonPhi.FillBoundary(geom.periodicity());
	
        // Calculate rho from Phi in SC region
        ComputeRho(PoissonPhi, charge_den, e_den, hole_den, geom);
        
	if (SC_hi[2] <= 0.) {
            // no semiconductor region; set error to zero so the while loop terminates
            err = 0.;
        } else {

            // Calculate Error
            if (iter > 0){
                MultiFab::Copy(PhiErr, PoissonPhi, 0, 0, 1, 0);
                MultiFab::Subtract(PhiErr, PoissonPhi_Prev, 0, 0, 1, 0);
                err = PhiErr.norm1(0, geom.periodicity())/PoissonPhi.norm1(0, geom.periodicity());
            }

            //Copy PoissonPhi to PoissonPhi_Prev to calculate error at the next iteration
            MultiFab::Copy(PoissonPhi_Prev, PoissonPhi, 0, 0, 1, 0);

            iter = iter + 1;
            amrex::Print() << iter << " iterations :: err = " << err << std::endl;
        }
    }
    
    amrex::Print() << "\n ========= Self-Consistent Initialization of P and Rho Done! ========== \n"<< iter << " iterations to obtain self consistent Phi with err = " << err << std::endl;

    // Calculate E from Phi
    ComputeEfromPhi(PoissonPhi, Ex, Ey, Ez, geom);

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,8);
        MultiFab::Copy(Plt, P_old[0], 0, 0, 1, 0);
        MultiFab::Copy(Plt, P_old[1], 0, 1, 1, 0);
        MultiFab::Copy(Plt, P_old[2], 0, 2, 1, 0);  
        MultiFab::Copy(Plt, PoissonPhi, 0, 3, 1, 0);
        MultiFab::Copy(Plt, PoissonRHS, 0, 4, 1, 0);
        MultiFab::Copy(Plt, Ex, 0, 5, 1, 0);
        MultiFab::Copy(Plt, Ey, 0, 6, 1, 0);
        MultiFab::Copy(Plt, Ez, 0, 7, 1, 0);
        MultiFab::Copy(Plt, hole_den, 0, 8, 1, 0);
        MultiFab::Copy(Plt, e_den, 0, 9, 1, 0);
        MultiFab::Copy(Plt, charge_den, 0, 10, 1, 0);
        MultiFab::Copy(Plt, beta_face[0], 0, 11, 1, 0);
        MultiFab::Copy(Plt, beta_face[1], 0, 12, 1, 0);
        MultiFab::Copy(Plt, beta_face[2], 0, 13, 1, 0);
        MultiFab::Copy(Plt, GL_rhs[2], 0, 14, 1, 0);
#ifdef AMREX_USE_EB
	amrex::EB_WriteSingleLevelPlotfile(pltfile, Plt, {"Px","Py","Pz","Phi","PoissonRHS","Ex","Ey","Ez","holes","electrons","charge","epsilon_xface","epsilon_yface","epsilon_zface"}, geom, time, 0);
#else
	amrex::WriteSingleLevelPlotfile(pltfile, Plt, {"Px","Py","Pz","Phi","PoissonRHS","Ex","Ey","Ez","holes","electrons","charge","epsilon_xface","epsilon_yface","epsilon_zface", "GL_rhs_z"}, geom, time, 0);
#endif
    }

    amrex::Print() << "\n ========= Advance Steps  ========== \n"<< std::endl;

    for (int step = 1; step <= nsteps; ++step)
    {
        Real step_strt_time = ParallelDescriptor::second();

        // compute f^n = f(P^n,Phi^n)
        CalculateTDGL_RHS(GL_rhs, P_old, PoissonPhi, Gamma, geom);

        // P^{n+1,*} = P^n + dt * f^n
        for (int i = 0; i < 3; i++){
            MultiFab::LinComb(P_new_pre[i], 1.0, P_old[i], 0, dt, GL_rhs[i], 0, 0, 1, Nghost);
            P_new_pre[i].FillBoundary(geom.periodicity()); 
        }  

	/**
         * \brief dst = a*x + b*y
         */
//    static void LinComb (MultiFab&       dst,
//                         Real            a,
//                         const MultiFab& x,
//                         int             xcomp,
//                         Real            b,
//                         const MultiFab& y,
//                         int             ycomp,
//                         int             dstcomp,
//                         int             numcomp,
//                         int             nghost);
	
        err = 1.0;
        iter = 0;

        // iterate to compute Phi^{n+1,*}
        //while(iter < 2){
        while(err > tol){
   
            // Compute RHS of Poisson equation
            ComputePoissonRHS(PoissonRHS, P_new_pre, charge_den, geom);

            dF_dPhi(alpha_cc, PoissonRHS, PoissonPhi, P_new_pre, charge_den, e_den, hole_den, geom);

            ComputePoissonRHS_Newton(PoissonRHS, PoissonPhi, alpha_cc); 

#ifdef AMREX_USE_EB
            p_mlebabec->setACoeffs(0, alpha_cc);
#else 
            p_mlabec->setACoeffs(0, alpha_cc);
#endif
            //Initial guess for phi
            PoissonPhi.setVal(0.);

            //Poisson Solve
            pMLMG->solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
	    
	    PoissonPhi.FillBoundary(geom.periodicity());
            
	    // Calculate rho from Phi in SC region
            ComputeRho(PoissonPhi, charge_den, e_den, hole_den, geom);

            if (SC_hi[2] <= 0.) {
                // no semiconductor region; set error to zero so the while loop terminates
                err = 0.;
            } else {
                
                // Calculate Error
                if (iter > 0){
                    MultiFab::Copy(PhiErr, PoissonPhi, 0, 0, 1, 0);
                    MultiFab::Subtract(PhiErr, PoissonPhi_Prev, 0, 0, 1, 0);
                    err = PhiErr.norm1(0, geom.periodicity())/PoissonPhi.norm1(0, geom.periodicity());
                }

                //Copy PoissonPhi to PoissonPhi_Prev to calculate error at the next iteration
                MultiFab::Copy(PoissonPhi_Prev, PoissonPhi, 0, 0, 1, 0);

                iter = iter + 1;
                amrex::Print() << iter << " iterations :: err = " << err << std::endl;
            }
        }
        
        if (TimeIntegratorOrder == 1) {

            // copy new solution into old solution
            for (int i = 0; i < 3; i++){
                MultiFab::Copy(P_old[i], P_new_pre[i], 0, 0, 1, 0);
                // fill periodic ghost cells
                P_old[i].FillBoundary(geom.periodicity());
            }
            
        } else {
        
            // compute f^{n+1,*} = f(P^{n+1,*},Phi^{n+1,*})
            CalculateTDGL_RHS(GL_rhs_pre, P_new_pre, PoissonPhi, Gamma, geom);

            // P^{n+1} = P^n + dt/2 * f^n + dt/2 * f^{n+1,*}
            for (int i = 0; i < 3; i++){
                MultiFab::LinComb(GL_rhs_avg[i], 0.5, GL_rhs[i], 0, 0.5, GL_rhs_pre[i], 0, 0, 1, Nghost);    
                MultiFab::LinComb(P_new[i], 1.0, P_old[i], 0, dt, GL_rhs_avg[i], 0, 0, 1, Nghost);
            }
        
            err = 1.0;
            iter = 0;

            // iterate to compute Phi^{n+1}
            //while(iter < 2){
            while(err > tol){
   
                // Compute RHS of Poisson equation
                ComputePoissonRHS(PoissonRHS, P_new, charge_den, geom);

                dF_dPhi(alpha_cc, PoissonRHS, PoissonPhi, P_new, charge_den, e_den, hole_den, geom);

                ComputePoissonRHS_Newton(PoissonRHS, PoissonPhi, alpha_cc); 

#ifdef AMREX_USE_EB
                p_mlebabec->setACoeffs(0, alpha_cc);
#else 
                p_mlabec->setACoeffs(0, alpha_cc);
#endif
 
                //Initial guess for phi
                PoissonPhi.setVal(0.);

                //Poisson Solve
                pMLMG->solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
	        PoissonPhi.FillBoundary(geom.periodicity());
	
                // Calculate rho from Phi in SC region
                ComputeRho(PoissonPhi, charge_den, e_den, hole_den, geom);

                if (SC_hi[2] <= 0.) {
                    // no semiconductor region; set error to zero so the while loop terminates
                    err = 0.;
                } else {
                    
                    // Calculate Error
                    if (iter > 0){
                        MultiFab::Copy(PhiErr, PoissonPhi, 0, 0, 1, 0);
                        MultiFab::Subtract(PhiErr, PoissonPhi_Prev, 0, 0, 1, 0);
                        err = PhiErr.norm1(0, geom.periodicity())/PoissonPhi.norm1(0, geom.periodicity());
                    }

                    //Copy PoissonPhi to PoissonPhi_Prev to calculate error at the next iteration
                    MultiFab::Copy(PoissonPhi_Prev, PoissonPhi, 0, 0, 1, 0);

                    iter = iter + 1;
                    amrex::Print() << iter << " iterations :: err = " << err << std::endl;
                }
            }
            
            // copy new solution into old solution
            for (int i = 0; i < 3; i++){
                MultiFab::Copy(P_old[i], P_new[i], 0, 0, 1, 0);
                // fill periodic ghost cells
                P_old[i].FillBoundary(geom.periodicity());
            }
        }

        if (inc_step > 0 && step%inc_step == 0) {
            Phi_Bc_hi = Phi_Bc_hi + Phi_Bc_inc;
            amrex::Print() << "step = " << step << ", Phi_Bc_hi = " << Phi_Bc_hi << std::endl;

            // Set Dirichlet BC for Phi in z
            //SetPhiBC_z(PoissonPhi);
    
            // set Dirichlet BC by reading in the ghost cell values
#ifdef AMREX_USE_EB
            p_mlebabec->setLevelBC(0, &PoissonPhi);
#else 
            p_mlabec->setLevelBC(0, &PoissonPhi);
#endif

            err = 1.0;
            iter = 0;

            // iterate to compute Phi^{n+1} with new Dirichlet value
            //while(iter < 10){
            while(err > tol){
   
                // Compute RHS of Poisson equation
                ComputePoissonRHS(PoissonRHS, P_new, charge_den, geom);

                dF_dPhi(alpha_cc, PoissonRHS, PoissonPhi, P_new, charge_den, e_den, hole_den, geom);

                ComputePoissonRHS_Newton(PoissonRHS, PoissonPhi, alpha_cc); 

#ifdef AMREX_USE_EB
                p_mlebabec->setACoeffs(0, alpha_cc);
#else 
                p_mlabec->setACoeffs(0, alpha_cc);
#endif
 
                //Initial guess for phi
                PoissonPhi.setVal(0.);

                //Poisson Solve
                pMLMG->solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
	        PoissonPhi.FillBoundary(geom.periodicity());
	
                // Calculate rho from Phi in SC region
                ComputeRho(PoissonPhi, charge_den, e_den, hole_den, geom);

                if (SC_hi[2] <= 0.) {
                    // no semiconductor region; set error to zero so the while loop terminates
                    err = 0.;
                } else {
                
                    // Calculate Error
                    if (iter > 0){
                        MultiFab::Copy(PhiErr, PoissonPhi, 0, 0, 1, 0);
                        MultiFab::Subtract(PhiErr, PoissonPhi_Prev, 0, 0, 1, 0);
                        err = PhiErr.norm1(0, geom.periodicity())/PoissonPhi.norm1(0, geom.periodicity());
                    }

                    //Copy PoissonPhi to PoissonPhi_Prev to calculate error at the next iteration
                    MultiFab::Copy(PoissonPhi_Prev, PoissonPhi, 0, 0, 1, 0);

                    iter = iter + 1;
                    amrex::Print() << iter << " iterations :: err = " << err << std::endl;
                }
            }
        }

        // Calculate E from Phi
	ComputeEfromPhi(PoissonPhi, Ex, Ey, Ez, geom);

	Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";
        amrex::Print() << " \n";

        // update time
        time = time + dt;

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,8);
            MultiFab::Copy(Plt, P_old[0], 0, 0, 1, 0);
            MultiFab::Copy(Plt, P_old[1], 0, 1, 1, 0);
            MultiFab::Copy(Plt, P_old[2], 0, 2, 1, 0);  
            MultiFab::Copy(Plt, PoissonPhi, 0, 3, 1, 0);
            MultiFab::Copy(Plt, PoissonRHS, 0, 4, 1, 0);
            MultiFab::Copy(Plt, Ex, 0, 5, 1, 0);
            MultiFab::Copy(Plt, Ey, 0, 6, 1, 0);
            MultiFab::Copy(Plt, Ez, 0, 7, 1, 0);
            MultiFab::Copy(Plt, hole_den, 0, 8, 1, 0);
            MultiFab::Copy(Plt, e_den, 0, 9, 1, 0);
            MultiFab::Copy(Plt, charge_den, 0, 10, 1, 0);
            MultiFab::Copy(Plt, beta_face[0], 0, 11, 1, 0);
            MultiFab::Copy(Plt, beta_face[1], 0, 12, 1, 0);
            MultiFab::Copy(Plt, beta_face[2], 0, 13, 1, 0);
            MultiFab::Copy(Plt, GL_rhs[2], 0, 14, 1, 0);
#ifdef AMREX_USE_EB
	    amrex::EB_WriteSingleLevelPlotfile(pltfile, Plt, {"Px","Py","Pz","Phi","PoissonRHS","Ex","Ey","Ez","holes","electrons","charge","epsilon_xface","epsilon_yface","epsilon_zface"}, geom, time, step);
#else
	    amrex::WriteSingleLevelPlotfile(pltfile, Plt, {"Px","Py","Pz","Phi","PoissonRHS","Ex","Ey","Ez","holes","electrons","charge","epsilon_xface","epsilon_yface","epsilon_zface", "GL_rhs_z"}, geom, time, step);
#endif
        }

    }

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
