
#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H> 
#include <AMReX_MultiFab.H> 
#include <AMReX_VisMF.H>
#include "myfunc.H"
#include "TDGL.H"

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc,argv);

    main_main();

    amrex::Finalize();
    return 0;
}

void main_main ()
{

    Real total_step_strt_time = ParallelDescriptor::second();

    // **********************************
    // SIMULATION PARAMETERS

    amrex::GpuArray<int, 3> n_cell; // Number of cells in each dimension

    // size of each box (or grid)
    int max_grid_size;

    // total steps in simulation
    int nsteps;

    // how often to write a plotfile
    int plot_int;

    // time step
    Real dt;
    
    amrex::GpuArray<amrex::Real, 3> prob_lo; // physical lo coordinate
    amrex::GpuArray<amrex::Real, 3> prob_hi; // physical hi coordinate

    int P_BC_flag_hi;
    int P_BC_flag_lo;
    Real Phi_Bc_hi;
    Real Phi_Bc_lo;
    Real Phi_Bc_inc;
    int inc_step;

    int TimeIntegratorOrder;

    // TDGL right hand side parameters
    Real epsilon_0, epsilonX_fe, epsilonZ_fe, epsilon_de, epsilon_si, alpha, beta, gamma, BigGamma, g11, g44;
    Real DE_lo, DE_hi, FE_lo, FE_hi, SC_lo, SC_hi;
    Real lambda;

    // inputs parameters
    {
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

        // Material Properties
	
        pp.get("epsilon_0",epsilon_0); // epsilon_0
        pp.get("epsilonX_fe",epsilonX_fe);// epsilon_r for FE
        pp.get("epsilonZ_fe",epsilonZ_fe);// epsilon_r for FE
        pp.get("epsilon_de",epsilon_de);// epsilon_r for DE
        pp.get("epsilon_si",epsilon_si);// epsilon_r for SC
        pp.get("alpha",alpha);
        pp.get("beta",beta);
        pp.get("gamma",gamma);
        pp.get("BigGamma",BigGamma);
        pp.get("g11",g11);
        pp.get("g44",g44);

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
    }


    // For Silicon:
    // Nc = 2.8e25 m^-3
    // Nv = 1.04e25 m^-3
    // Ec = 1.12eV and Ev = 0, such that band gap Eg = 1.12eV
    // 1eV = 1.602e-19 J

    Real Nc = 2.8e25;
    Real Nv = 1.04e25;
    Real Ec = 0.56;//*1.602e-19;
    Real Ev = -0.56;//*1.602e-19;
    Real q = 1.602e-19; 
    Real kb = 1.38e-23; // Boltzmann constant
    Real T = 300; // Room Temp

    // **********************************
    // SIMULATION SETUP

    // make BoxArray and Geometry
    // ba will contain a list of boxes that cover the domain
    // geom contains information such as the physical domain size,
    //               number of points in the domain, and periodicity
    BoxArray ba;
    Geometry geom;

    // AMREX_D_DECL means "do the first X of these, where X is the dimensionality of the simulation"
    IntVect dom_lo(AMREX_D_DECL(       0,        0,        0));
    IntVect dom_hi(AMREX_D_DECL(n_cell[0]-1, n_cell[1]-1, n_cell[2]-1));

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    ba.define(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    // This defines the physical box in each direction.
    RealBox real_box({AMREX_D_DECL( prob_lo[0], prob_lo[1], prob_lo[2])},
                     {AMREX_D_DECL( prob_hi[0], prob_hi[1], prob_hi[2])});

    // periodic in x and y directions
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,0)};

    // This defines a Geometry object
    geom.define(domain, real_box, CoordSys::cartesian, is_periodic);

    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // Nghost = number of ghost cells for each array
    int Nghost = 1;

    // Ncomp = number of components for each array
    int Ncomp = 1;

    // How Boxes are distrubuted among MPI processes
    DistributionMapping dm(ba);

    // we allocate two P multifabs; one will store the old state, the other the new.
    MultiFab P_old(ba, dm, Ncomp, Nghost);
    MultiFab P_new(ba, dm, Ncomp, Nghost);
    MultiFab P_new_pre(ba, dm, Ncomp, Nghost);
    MultiFab Gamma(ba, dm, Ncomp, Nghost);
    MultiFab GL_rhs(ba, dm, Ncomp, Nghost);
    MultiFab GL_rhs_pre(ba, dm, Ncomp, Nghost);
    MultiFab GL_rhs_avg(ba, dm, Ncomp, Nghost);
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

    MultiFab Plt(ba, dm, 9, 0);
    MultiFab Plt_debug(ba, dm, 4, 0);

    //Solver for Poisson equation
    LPInfo info;
    MLABecLaplacian mlabec({geom}, {ba}, {dm}, info);

    //Force singular system to be solvable
    mlabec.setEnforceSingularSolvable(false); 

    // order of stencil
    int linop_maxorder = 2;
    mlabec.setMaxOrder(linop_maxorder);  

    // build array of boundary conditions needed by MLABecLaplacian
    std::array<LinOpBCType, AMREX_SPACEDIM> lo_mlmg_bc;
    std::array<LinOpBCType, AMREX_SPACEDIM> hi_mlmg_bc; 

    //Periodic 
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if(is_periodic[idim]){
          lo_mlmg_bc[idim] = hi_mlmg_bc[idim] = LinOpBCType::Periodic;
        } else {
          lo_mlmg_bc[idim] = hi_mlmg_bc[idim] = LinOpBCType::Dirichlet;
        }
    } 

    mlabec.setDomainBC(lo_mlmg_bc,hi_mlmg_bc);

    // coefficients for solver
    MultiFab alpha_cc(ba, dm, 1, 0);
    std::array< MultiFab, AMREX_SPACEDIM > beta_face;
    AMREX_D_TERM(beta_face[0].define(convert(ba,IntVect(AMREX_D_DECL(1,0,0))), dm, 1, 0);,
                 beta_face[1].define(convert(ba,IntVect(AMREX_D_DECL(0,1,0))), dm, 1, 0);,
                 beta_face[2].define(convert(ba,IntVect(AMREX_D_DECL(0,0,1))), dm, 1, 0););
    
    // set cell-centered alpha coefficient to zero
    alpha_cc.setVal(0.);

    // set face-centered beta coefficient to 
    // epsilon values in SC, FE, and DE layers
    InitializePermittivity(beta_face,
		        FE_lo, FE_hi, DE_lo, DE_hi, SC_lo, SC_hi,
                        epsilon_0, epsilonX_fe, epsilonZ_fe, epsilon_de, epsilon_si,
                        prob_lo, prob_hi,
                        geom);

    // Set Dirichlet BC for Phi in z
    SetPhiBC_z(PoissonPhi, n_cell, Phi_Bc_lo, Phi_Bc_hi); 
    
    // set Dirichlet BC by reading in the ghost cell values
    mlabec.setLevelBC(0, &PoissonPhi);
    
    // (A*alpha_cc - B * div beta grad) phi = rhs
    mlabec.setScalars(0.0, 1.0); // A = 0.0, B = 1.0
    mlabec.setACoeffs(0, alpha_cc); //First argument 0 is lev
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(beta_face));  

    //Declare MLMG object
    MLMG mlmg(mlabec);
//    mlmg.setVerbose(2);

    // time = starting time in the simulation
    Real time = 0.0;

    // INITIALIZE P in FE and rho in SC regions

    InitializePandRho(P_old, Gamma, charge_den, e_den, hole_den, 
		    SC_lo, SC_hi, DE_lo, DE_hi, 
		    BigGamma, q, Ec, Ev, kb, T, Nc, Nv,
		    prob_lo, prob_hi, 
		    geom);

    //Obtain self consisten Phi and rho
    Real tol = 1.e-5;
    Real err = 1.0;
    int iter = 0;
    //while(iter < 10){
    while(err > tol){
   
	//Compute RHS of Poisson equation
	ComputePoissonRHS(PoissonRHS, P_old, charge_den, 
			FE_lo, FE_hi, DE_lo, DE_hi, SC_lo, SC_hi, 
			P_BC_flag_lo, P_BC_flag_hi, 
			lambda, 
			prob_lo, prob_hi, 
			geom);
        // Set Dirichlet BC for Phi in z
        SetPhiBC_z(PoissonPhi, n_cell, Phi_Bc_lo, Phi_Bc_hi); 
        
        // set Dirichlet BC by reading in the ghost cell values
        mlabec.setLevelBC(0, &PoissonPhi);
        
        //Declare MLMG object
        MLMG mlmg(mlabec);

        //Initial guess for phi
        PoissonPhi.setVal(0.);
        //Poisson Solve
        mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);
	
        // Calculate rho from Phi in SC region

        ComputeRho(PoissonPhi, charge_den, e_den, hole_den, 
                   SC_lo, SC_hi,
                   q, Ec, Ev, kb, T, Nc, Nv,
                   prob_lo, prob_hi, geom);

	// Calculate Error

	if (iter > 0){
	   MultiFab::Copy(PhiErr, PoissonPhi, 0, 0, 1, 0);
	   MultiFab::Subtract(PhiErr, PoissonPhi_Prev, 0, 0, 1, 0);
	   err = PhiErr.norm1(0, geom.periodicity())/PoissonPhi.norm1(0, geom.periodicity());
        }

	//Copy PoissonPhi to PoissonPhi_Prev to calculate error at the next iteration
	
        MultiFab::Copy(PoissonPhi_Prev, PoissonPhi, 0, 0, 1, 0);

	iter = iter + 1;
        std::cout << iter << " iterations :: err = " << err << std::endl;
    }
    
    std::cout << "\n ========= Self-Consistent Initialization of P and Rho Done! ========== \n"<< iter << " iterations to obtain self consistent Phi with err = " << err << std::endl;
    std::cout << "\n ========= Advance Steps  ========== \n"<< std::endl;

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,8);
        MultiFab::Copy(Plt, P_old, 0, 0, 1, 0);  
        MultiFab::Copy(Plt, PoissonPhi, 0, 1, 1, 0);
        MultiFab::Copy(Plt, PoissonRHS, 0, 2, 1, 0);
        MultiFab::Copy(Plt, Ex, 0, 3, 1, 0);
        MultiFab::Copy(Plt, Ey, 0, 4, 1, 0);
        MultiFab::Copy(Plt, Ez, 0, 5, 1, 0);
        MultiFab::Copy(Plt, hole_den, 0, 6, 1, 0);
        MultiFab::Copy(Plt, e_den, 0, 7, 1, 0);
        MultiFab::Copy(Plt, charge_den, 0, 8, 1, 0);
        WriteSingleLevelPlotfile(pltfile, Plt, {"P","Phi","PoissonRHS","Ex","Ey","Ez","holes","electrons","charge"}, geom, time, 0);
    }

    for (int step = 1; step <= nsteps; ++step)
    {
        Real step_strt_time = ParallelDescriptor::second();

    	    // Evolve P
	
	CalculateTDGL_RHS(GL_rhs, P_old, PoissonPhi, Gamma, 
			FE_lo, FE_hi, DE_lo, DE_hi, SC_lo, SC_hi, 
			P_BC_flag_lo, P_BC_flag_hi, Phi_Bc_lo, Phi_Bc_hi, 
			alpha, beta, gamma, g11, g44, lambda, 
			prob_lo, prob_hi, 
			geom);

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
	

	if(TimeIntegratorOrder == 1)
        //1st Order Forward Euler
	{
        	MultiFab::LinComb(P_new, 1.0, P_old, 0, dt, GL_rhs, 0, 0, 1, Nghost);   
	} else
	{	
        //2nd Order Predictor-Corrector

	        //Predictor
	        MultiFab::LinComb(P_new_pre, 1.0, P_old, 0, dt, GL_rhs, 0, 0, 1, Nghost);    
	        
	        //New RHS	
	        ComputePoissonRHS(PoissonRHS, P_new_pre, charge_den, 
	        		FE_lo, FE_hi, DE_lo, DE_hi, SC_lo, SC_hi, 
	        		P_BC_flag_lo, P_BC_flag_hi, lambda, 
	        		prob_lo, prob_hi, 
	        		geom);

                PoissonPhi.setVal(0.);
                mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);

	        CalculateTDGL_RHS(GL_rhs_pre, P_new_pre, PoissonPhi, Gamma, 
	        		FE_lo, FE_hi, DE_lo, DE_hi, SC_lo, SC_hi, 
	        		P_BC_flag_lo, P_BC_flag_hi, Phi_Bc_lo, Phi_Bc_hi, 
	        		alpha, beta, gamma, g11, g44, lambda, 
	        		prob_lo, prob_hi, 
	        		geom);

	        //Get average of GL_rhs and GL_rhs_pre
	        MultiFab::LinComb(GL_rhs_avg, 0.5, GL_rhs, 0, 0.5, GL_rhs_pre, 0, 0, 1, Nghost);    
	        //Corrector
	        MultiFab::LinComb(P_new, 1.0, P_old, 0, dt, GL_rhs_avg, 0, 0, 1, Nghost);

        }
        // copy new solution into old solution
        MultiFab::Copy(P_old, P_new, 0, 0, 1, 0);

        // fill periodic ghost cells
        P_old.FillBoundary(geom.periodicity());

	//Compute RHS of Poisson equation
	ComputePoissonRHS(PoissonRHS, P_old, charge_den, 
			FE_lo, FE_hi, DE_lo, DE_hi, SC_lo, SC_hi, 
			P_BC_flag_lo, P_BC_flag_hi, lambda, 
			prob_lo, prob_hi, 
			geom);

        if (inc_step > 0 && step%inc_step == 0) {
            Phi_Bc_hi = Phi_Bc_hi + Phi_Bc_inc;
        }
        amrex::Print() << "step = " << step << ", Phi_Bc_hi = " << Phi_Bc_hi << std::endl;
        
        // Set Dirichlet BC for Phi in z
        SetPhiBC_z(PoissonPhi, n_cell, Phi_Bc_lo, Phi_Bc_hi); 
        
        // set Dirichlet BC by reading in the ghost cell values
        mlabec.setLevelBC(0, &PoissonPhi);
        
        //Declare MLMG object
        MLMG mlmg(mlabec);

        //Initial guess for phi
        PoissonPhi.setVal(0.);
        mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1);

        // Calculate rho from Phi in SC region

        ComputeRho(PoissonPhi, charge_den, e_den, hole_den, 
                   SC_lo, SC_hi,
                   q, Ec, Ev, kb, T, Nc, Nv,
                   prob_lo, prob_hi, geom);


        // Calculate E from Phi

	ComputeEfromPhi(PoissonPhi, Ex, Ey, Ez, prob_lo, prob_hi, geom);

	Real step_stop_time = ParallelDescriptor::second() - step_strt_time;
        ParallelDescriptor::ReduceRealMax(step_stop_time);

        amrex::Print() << "Advanced step " << step << " in " << step_stop_time << " seconds\n";

        // update time
        time = time + dt;

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,8);
            MultiFab::Copy(Plt, P_old, 0, 0, 1, 0);  
            MultiFab::Copy(Plt, PoissonPhi, 0, 1, 1, 0);
            MultiFab::Copy(Plt, PoissonRHS, 0, 2, 1, 0);
            MultiFab::Copy(Plt, Ex, 0, 3, 1, 0);
            MultiFab::Copy(Plt, Ey, 0, 4, 1, 0);
            MultiFab::Copy(Plt, Ez, 0, 5, 1, 0);
            MultiFab::Copy(Plt, hole_den, 0, 6, 1, 0);
            MultiFab::Copy(Plt, e_den, 0, 7, 1, 0);
            MultiFab::Copy(Plt, charge_den, 0, 8, 1, 0);
            WriteSingleLevelPlotfile(pltfile, Plt, {"P","Phi","PoissonRHS","Ex","Ey","Ez","holes","electrons","charge"}, geom, time, step);
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

    }
    
        Real total_step_stop_time = ParallelDescriptor::second() - total_step_strt_time;
        ParallelDescriptor::ReduceRealMax(total_step_stop_time);

        amrex::Print() << "Total run time " << total_step_stop_time << " seconds\n";
}
