#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H> 
#include <AMReX_VisMF.H>
#include "myfunc.H"

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
        pp.get("P_BC_flag_lo",P_BC_flag_hi); // 0 : P = 0, 1 : dp/dz = p/lambda, 2 : dp/dz = 0
        pp.get("Phi_Bc_hi",Phi_Bc_hi);
        pp.get("Phi_Bc_lo",Phi_Bc_lo);

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
    Real Ec = 1.12*1.602e-19;
    Real Ev = 0.00*1.602e-19;
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
    MultiFab Gamma(ba, dm, Ncomp, Nghost);
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

    MultiFab Plt(ba, dm, 8, 0);
    MultiFab Plt_debug(ba, dm, 3, 0);

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
    // loop over boxes
    for (MFIter mfi(beta_face[0]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& beta_f0 = beta_face[0].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          Real z = (k+0.5) * dx[2];
          if(z <= SC_hi) {
             beta_f0(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  } else if(z <= DE_hi) {
             beta_f0(i,j,k) = epsilon_de * epsilon_0; //DE layer
          } else {
             beta_f0(i,j,k) = epsilonX_fe * epsilon_0; //FE layer
          }
        });
    }
    
    for (MFIter mfi(beta_face[1]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& beta_f1 = beta_face[1].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          Real z = (k+0.5) * dx[2];
          if(z <= SC_hi) {
             beta_f1(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  } else if(z <= DE_hi) {
            beta_f1(i,j,k) = epsilon_de * epsilon_0; //DE layer
          } else {
            beta_f1(i,j,k) = epsilonX_fe * epsilon_0; //FE layer
          }
        });
    }
    
    for (MFIter mfi(beta_face[2]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& beta_f2 = beta_face[2].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          Real z = k * dx[2];
          if(z <= SC_hi) {
             beta_f2(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  } else if(z <= DE_hi) {
            beta_f2(i,j,k) = epsilon_de * epsilon_0; //DE layer
          } else {
            beta_f2(i,j,k) = epsilonZ_fe * epsilon_0; //FE layer
          }
        });
    }
    
    // Set Dirichlet BC for Phi in z 
    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(1);

        const Array4<Real>& Phi = PoissonPhi.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          if(k < 0) {
            Phi(i,j,k) = Phi_Bc_lo;
          } else if(k >= n_cell[2]){
            Phi(i,j,k) = Phi_Bc_hi;
          }
        });
    }
    
    // set Dirichlet BC by reading in the ghost cell values
    mlabec.setLevelBC(0, &PoissonPhi);
    
    // (A*alpha_cc - B * div beta grad) phi = rhs
    mlabec.setScalars(0.0, 1.0); // A = 0.0, B = 1.0
    mlabec.setACoeffs(0, alpha_cc); //First argument 0 is lev
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(beta_face));  

    // time = starting time in the simulation
    Real time = 0.0;

    // **********************************
    // INITIALIZE P and Gamma such that it is zero in DE region

    // loop over boxes
    for (MFIter mfi(P_old); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& pOld = P_old.array(mfi);
        const Array4<Real>& Gam = Gamma.array(mfi);

        // set P
        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept 
        {
            Real x = (i+0.5) * dx[0];
            Real y = (j+0.5) * dx[1];
            Real z = (k+0.5) * dx[2];
            if (z <= DE_hi) {
               pOld(i,j,k) = 0.0;
               Gam(i,j,k) = 0.0;
            } else {
	       //double tmp = (i%3 + j%2 + k%4)/6.;
	       //pOld(i,j,k) = (-1.0 + 2.0*tmp)*0.002;
	       pOld(i,j,k) = (-1.0 + 2.0*Random())*0.002;
               Gam(i,j,k) = BigGamma;
            }
        });
    }

    // INITIALIZE rho in SC region

    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev 

        const Array4<Real>& hole_den_arr = hole_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = charge_den.array(mfi);
        const Array4<Real>& phi = PoissonPhi.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
             Real z = (k+0.5) * dx[2];

	     if(z <= SC_hi){ //SC region

                hole_den_arr(i,j,k) = Nv*exp(-0.00*1.602e-19/(kb*T)); // Testing phi = 0 initialization
                e_den_arr(i,j,k) = Nc*exp(-1.12*1.602e-19/(kb*T)); // Testing phi = 0 initialization
	        charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k)); // Testing phi = 0 initialization
                //charge_den_arr(i,j,k) = 0.0; // Testing rho = 0 initialization
             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
    
    //Obtain self consisten Phi and rho
    Real tol = 1.e-2;
    Real err = 1.0;
    int iter = 0;
    while(iter < 10){
    //while(err > tol){
    
        for ( MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real>& pOld = P_old.array(mfi);
            const Array4<Real>& RHS = PoissonRHS.array(mfi);
            const Array4<Real>& charge_den_arr = charge_den.array(mfi);
            const Array4<Real>& phi = PoissonPhi.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                 Real z = (k+0.5) * dx[2];
                 Real z_hi = (k+1.5) * dx[2];
                 Real z_lo = (k-0.5) * dx[2];

		 if(z <= SC_hi){ //SC region

		   RHS(i,j,k) = charge_den_arr(i,j,k);

		 } else if(z < DE_hi){ //DE region

		   RHS(i,j,k) = 0.;

		 } else if (DE_hi > z_lo && DE_hi <= z) { //FE side of FE-DE interface

                   if(P_BC_flag_lo == 0){
                     Real P_int = 0.0; 
                     RHS(i,j,k) = -(-4.*P_int + 3.*pOld(i,j,k) + pOld(i,j,k+1))/(3.*dx[2]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k+1)
                   } else if (P_BC_flag_lo == 1){
                     Real P_int = pOld(i,j,k)/(1 + dx[2]/2/lambda);
		     Real dPdz = P_int/lambda; 
                     RHS(i,j,k) = -(dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/(2.*dx[2]);
                   } else if (P_BC_flag_lo == 2){
		     Real dPdz = 0.; 
                     RHS(i,j,k) = -(dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/(2.*dx[2]);
                   }

                 } else if (z_hi > prob_hi[2]){ //Top metal

                   if(P_BC_flag_hi == 0){
                     Real P_int = 0.0; 
                     RHS(i,j,k) = -(4.*P_int - 3.*pOld(i,j,k) - pOld(i,j,k-1))/(3.*dx[2]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k-1)
                     } else if (P_BC_flag_hi == 1){
                     Real P_int = pOld(i,j,k)/(1 - dx[2]/2/lambda); 
		     Real dPdz = P_int/lambda; 
                     RHS(i,j,k) = -(dx[2]*dPdz + pOld(i,j,k) - pOld(i,j,k-1))/(2.*dx[2]);
                     } else if (P_BC_flag_hi == 2){
		     Real dPdz = 0.; 
                     RHS(i,j,k) = -(dx[2]*dPdz + pOld(i,j,k) - pOld(i,j,k-1))/(2.*dx[2]);
                   }

                 }else{ //inside FE

                   RHS(i,j,k) = -(pOld(i,j,k+1) - pOld(i,j,k-1))/(2.*dx[2]);

                 }

            });
        }

        //Initial guess for phi
        PoissonPhi.setVal(0.);

        MLMG mlmg(mlabec);
        mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1); //1e-10 for rel_tol and -1 (to ignore)

	//VisMF::Write(PoissonPhi,"Phi_init");
	//VisMF::Write(PoissonRHS,"RHS_init");
        //amrex::Abort("Abort here.");
`	
        // Calculate rho from Phi in SC region

        for ( MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            // Calculate charge density from Phi, Nc, Nv, Ec, and Ev 

            const Array4<Real>& hole_den_arr = hole_den.array(mfi);
            const Array4<Real>& e_den_arr = e_den.array(mfi);
            const Array4<Real>& charge_den_arr = charge_den.array(mfi);
            const Array4<Real>& phi = PoissonPhi.array(mfi);

            amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                 Real z = (k+0.5) * dx[2];

		 if(z <= SC_hi){ //SC region

                    hole_den_arr(i,j,k) = Nv*exp(-(q*phi(i,j,k) - Ev)/(kb*T));
                    e_den_arr(i,j,k) = Nc*exp(-(Ec - q*phi(i,j,k))/(kb*T));
		    charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));
	            if(i == 5 && j == 5)std::cout <<" Ec - q*phi = " << (Ec - (q*phi(i,j,k)))*6.242e+18 << " qphi - Ev = " << (q*phi(i,j,k) - Ev)*6.242e+18 << " hole = " << hole_den_arr(i,j,k) << ", e_den = " << e_den_arr(i,j,k) << ", charge_den = " << charge_den_arr(i,j,k) << std::endl;
	            //if(i == 5 && j == 5 && k == 5)std::cout << "hole = " << hole_den_arr(i,j,k) << ", e_den = " << e_den_arr(i,j,k) << ", charge_den = " << charge_den_arr(i,j,k) << std::endl;
                 } else {

                    charge_den_arr(i,j,k) = 0.0;

                 }
             });
        }

	// Calculate Error

	if (iter > 0){
	   MultiFab::Copy(PhiErr, PoissonPhi, 0, 0, 1, 0);
	   MultiFab::Subtract(PhiErr, PoissonPhi_Prev, 0, 0, 1, 0);
	   err = PhiErr.norm1(0, geom.periodicity())/PoissonPhi.norm1(0, geom.periodicity());
        }

        const std::string& pltfile = amrex::Concatenate("plt_debug",iter,0);
        MultiFab::Copy(Plt_debug, hole_den, 0, 0, 1, 0);
        MultiFab::Copy(Plt_debug, e_den, 0, 1, 1, 0);
        MultiFab::Copy(Plt_debug, charge_den, 0, 2, 1, 0);
        WriteSingleLevelPlotfile(pltfile, Plt_debug, {"holes","electrons","charge"}, geom, time, iter);
	//Copy PoissonPhi to PoissonPhi_Prev to calculate error at the next iteration
	
        MultiFab::Copy(PoissonPhi_Prev, PoissonPhi, 0, 0, 1, 0);

	iter = iter + 1;
        std::cout << iter << " iterations :: err = " << err << std::endl;
    }
    
    std::cout << iter << " iterations to obtain self consistent Phi with err = " << err << std::endl;

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,8);
        MultiFab::Copy(Plt, P_old, 0, 0, 1, 0);  
        MultiFab::Copy(Plt, PoissonPhi, 0, 1, 1, 0);
        MultiFab::Copy(Plt, Ex, 0, 2, 1, 0);
        MultiFab::Copy(Plt, Ey, 0, 3, 1, 0);
        MultiFab::Copy(Plt, Ez, 0, 4, 1, 0);
        MultiFab::Copy(Plt, hole_den, 0, 5, 1, 0);
        MultiFab::Copy(Plt, e_den, 0, 6, 1, 0);
        MultiFab::Copy(Plt, charge_den, 0, 7, 1, 0);
        WriteSingleLevelPlotfile(pltfile, Plt, {"P","Phi","Ex","Ey","Ez","holes","electrons","charge"}, geom, time, 0);
    }

    for (int step = 1; step <= nsteps; ++step)
    {
        // fill periodic ghost cells
        P_old.FillBoundary(geom.periodicity());

        // Calculate right hand side 

        for ( MFIter mfi(P_old); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real>& pOld = P_old.array(mfi);
            const Array4<Real>& RHS = PoissonRHS.array(mfi);
            const Array4<Real>& charge_den_arr = charge_den.array(mfi);

            // advance the data by dt
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                 Real z = (k+0.5) * dx[2];
                 Real z_hi = (k+1.5) * dx[2];
                 Real z_lo = (k-0.5) * dx[2];

		 if(z <= SC_hi){ //SC region

		   RHS(i,j,k) = charge_den_arr(i,j,k);

		 } else if(z < DE_hi){ //DE region

		   RHS(i,j,k) = 0.;
		 } else if (DE_hi > z_lo && DE_hi <= z) { //FE side of FE-DE interface

                   if(P_BC_flag_lo == 0){
                     Real P_int = 0.0; 
                     RHS(i,j,k) = -(-4.*P_int + 3.*pOld(i,j,k) + pOld(i,j,k+1))/(3.*dx[2]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k+1)
                   } else if (P_BC_flag_lo == 1){
                     Real P_int = pOld(i,j,k)/(1 + dx[2]/2/lambda);
		     Real dPdz = P_int/lambda; 
                     RHS(i,j,k) = -(dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/(2.*dx[2]);
                   } else if (P_BC_flag_lo == 2){
		     Real dPdz = 0.; 
                     RHS(i,j,k) = -(dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/(2.*dx[2]);
                   }

                 } else if (z_hi > prob_hi[2]){ //Top metal

                   if(P_BC_flag_hi == 0){
                     Real P_int = 0.0; 
                     RHS(i,j,k) = -(4.*P_int - 3.*pOld(i,j,k) - pOld(i,j,k-1))/(3.*dx[2]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k-1)
                     } else if (P_BC_flag_hi == 1){
                     Real P_int = pOld(i,j,k)/(1 - dx[2]/2/lambda); 
		     Real dPdz = P_int/lambda; 
                     RHS(i,j,k) = -(dx[2]*dPdz + pOld(i,j,k) - pOld(i,j,k-1))/(2.*dx[2]);
                     } else if (P_BC_flag_hi == 2){
		     Real dPdz = 0.; 
                     RHS(i,j,k) = -(dx[2]*dPdz + pOld(i,j,k) - pOld(i,j,k-1))/(2.*dx[2]);
                   }

                 }else{ //inside FE

                   RHS(i,j,k) = -(pOld(i,j,k+1) - pOld(i,j,k-1))/(2.*dx[2]);

                 }

            });
        }

        //Initial guess for phi
        PoissonPhi.setVal(0.);

        MLMG mlmg(mlabec);
        mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1); //1e-10 for rel_tol and -1 (to ignore) 
        //mlmg.solve({&PoissonPhi}, {&PoissonRHS}, mg_rel_tol, mg_abs_tol); //1e-10 for rel_tol and -1 (to ignore) 

        // Calculate E from Phi

        for ( MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real>& Ex_arr = Ex.array(mfi);
            const Array4<Real>& Ey_arr = Ey.array(mfi);
            const Array4<Real>& Ez_arr = Ez.array(mfi);
            const Array4<Real>& phi = PoissonPhi.array(mfi);

            amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                     Ex_arr(i,j,k) = -(phi(i+1,j,k) - phi(i-1,j,k))/(2.*dx[0]);
                     Ey_arr(i,j,k) = -(phi(i,j+1,k) - phi(i,j-1,k))/(2.*dx[1]);
          
                     Real z = (k+0.5) * dx[2];
                     Real z_hi = (k+1.5) * dx[2];
                     Real z_lo = (k-0.5) * dx[2];

	             if(z_lo < prob_lo[2]){ //Bottom Boundary
                       Ez_arr(i,j,k) = -(phi(i,j,k+1) - phi(i,j,k))/(dx[2]);
                     } else if (z_hi > prob_hi[2]){ //Top Boundary
                       Ez_arr(i,j,k) = -(phi(i,j,k) - phi(i,j,k-1))/(dx[2]);
                     }else{ //inside
                       Ez_arr(i,j,k) = -(phi(i,j,k+1) - phi(i,j,k-1))/(2.*dx[2]);
                     }
             });
        }

        // Calculate rho from Phi in SC region

        for ( MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            // Calculate charge density from Phi, Nc, Nv, Ec, and Ev 

            const Array4<Real>& hole_den_arr = hole_den.array(mfi);
            const Array4<Real>& e_den_arr = e_den.array(mfi);
            const Array4<Real>& charge_den_arr = charge_den.array(mfi);
            const Array4<Real>& phi = PoissonPhi.array(mfi);

            amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                 Real z = (k+0.5) * dx[2];

		 if(z <= SC_hi){ //SC region

                    hole_den_arr(i,j,k) = Nv*exp(-(q*phi(i,j,k) - Ev)/(kb*T));
                    e_den_arr(i,j,k) = Nc*exp(-(Ec - q*phi(i,j,k))/(kb*T));
		    charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));
	            //if(i == 5 && j == 5 && k == 5)std::cout << "hole = " << hole_den_arr(i,j,k) << ", e_den = " << e_den_arr(i,j,k) << ", charge_den = " << charge_den_arr(i,j,k) << std::endl;
                 } else {

                    charge_den_arr(i,j,k) = 0.0;

                 }
             });
        }


        // Evolve P
        // loop over boxes
        for ( MFIter mfi(P_old); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real>& pOld = P_old.array(mfi);
            const Array4<Real>& pNew = P_new.array(mfi);
            const Array4<Real>& phi = PoissonPhi.array(mfi);
            const Array4<Real>& Gam = Gamma.array(mfi);


            // advance the data by dt
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real grad_term, phi_term, d2P_z;
                Real z = (k+0.5) * dx[2];
                Real z_hi = (k+1.5) * dx[2];
                Real z_lo = (k-0.5) * dx[2];

		if(z_lo < prob_lo[2]){ //Bottom metal

                  grad_term = 0.0;
                  phi_term = (-4.*Phi_Bc_lo + 3.*phi(i,j,k) + phi(i,j,k+1))/(3.*dx[2]);
                  //phi_term = (phi(i,j,k+1) - phi(i,j,k)) / (dx[2]);

		} else if(z < DE_hi){ //Below FE-DE interface

                  grad_term = 0.0;
                  phi_term = (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2]);

		} else if (DE_hi > z_lo && DE_hi <= z) { //FE side of FE-DE interface

                  if(P_BC_flag_lo == 0){
                    Real P_int = 0.0;
		    d2P_z = 4.*(2.*P_int - 3.*pOld(i,j,k) + pOld(i,j,k+1))/3./dx[2]/dx[2];//2nd Order 
                  } else if (P_BC_flag_lo == 1){
                    Real P_int = pOld(i,j,k)/(1 + dx[2]/2/lambda);
		    Real dPdz = P_int/lambda; 
		    d2P_z = (-dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/dx[2]/dx[2];//2nd Order 
                  } else if (P_BC_flag_lo == 2){
		    Real dPdz = 0.; 
		    d2P_z = (-dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/dx[2]/dx[2];//2nd Order 
                  }

		  grad_term = g11 * d2P_z;
                  phi_term = (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2]);

                } else if (z_hi > prob_hi[2]){ //Top metal

                  if(P_BC_flag_hi == 0){
                    Real P_int = 0.0; 
		    d2P_z = 4.*(2.*P_int - 3.*pOld(i,j,k) + pOld(i,j,k-1))/3./dx[2]/dx[2];//2nd Order 
                  } else if (P_BC_flag_hi == 1){
                    Real P_int = pOld(i,j,k)/(1 - dx[2]/2/lambda); 
                    Real dPdz = P_int/lambda;
		    d2P_z = (dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k-1))/dx[2]/dx[2];//2nd Order 
                  } else if (P_BC_flag_hi == 2){
		    Real dPdz = 0.; 
		    d2P_z = (dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k-1))/dx[2]/dx[2];//2nd Order 
                  }

		  grad_term = g11 * d2P_z;
                  phi_term = (4.*Phi_Bc_hi - 3.*phi(i,j,k) - phi(i,j,k-1))/(3.*dx[2]);

                }else{ //inside FE

                  grad_term = g11 * (pOld(i,j,k+1) - 2.*pOld(i,j,k) + pOld(i,j,k-1)) / (dx[2]*dx[2]);
                  phi_term = (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2]);

                }

                pNew(i,j,k) = pOld(i,j,k) - dt * Gam(i,j,k) *
                    (  alpha*pOld(i,j,k) + beta*std::pow(pOld(i,j,k),3.) + gamma*std::pow(pOld(i,j,k),5.)
                     - g44 * (pOld(i+1,j,k) - 2.*pOld(i,j,k) + pOld(i-1,j,k)) / (dx[0]*dx[0])
                     - g44 * (pOld(i,j+1,k) - 2.*pOld(i,j,k) + pOld(i,j-1,k)) / (dx[1]*dx[1])
                     - grad_term
                     + phi_term
                    );
            });
        }

        // update time
        time = time + dt;

        // copy new solution into old solution
        MultiFab::Copy(P_old, P_new, 0, 0, 1, 0);

        // Tell the I/O Processor to write out which step we're doing
        amrex::Print() << "Advanced step " << step << "\n";

        // Write a plotfile of the current data (plot_int was defined in the inputs file)
        if (plot_int > 0 && step%plot_int == 0)
        {
            const std::string& pltfile = amrex::Concatenate("plt",step,8);
            MultiFab::Copy(Plt, P_old, 0, 0, 1, 0);  
            MultiFab::Copy(Plt, PoissonPhi, 0, 1, 1, 0);
            MultiFab::Copy(Plt, Ex, 0, 2, 1, 0);
            MultiFab::Copy(Plt, Ey, 0, 3, 1, 0);
            MultiFab::Copy(Plt, Ez, 0, 4, 1, 0);
            MultiFab::Copy(Plt, hole_den, 0, 5, 1, 0);
            MultiFab::Copy(Plt, e_den, 0, 6, 1, 0);
            MultiFab::Copy(Plt, charge_den, 0, 7, 1, 0);
            WriteSingleLevelPlotfile(pltfile, Plt, {"P","Phi","Ex","Ey","Ez","holes","electrons","charge"}, geom, time, step);
        }
    }
}
