#include <AMReX_PlotFileUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_MLMG.H> 

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

    // number of cells on each side of the domain
    int n_cell;

    // size of each box (or grid)
    int max_grid_size;

    // TDGL right hand side parameters
    Real epsilon, alpha, beta, gamma, BigGamma, g11, g44;

    // total steps in simulation
    int nsteps;

    // how often to write a plotfile
    int plot_int;

    // time step
    Real dt;
    
    amrex::GpuArray<amrex::Real, 3> prob_lo; // physical lo coordinate
    amrex::GpuArray<amrex::Real, 3> prob_hi; // physical hi coordinate

    // inputs parameters
    {
        // ParmParse is way of reading inputs from the inputs file
        // pp.get means we require the inputs file to have it
        // pp.query means we optionally need the inputs file to have it - but we must supply a default here
        ParmParse pp;

        // We need to get n_cell from the inputs file - this is the number of cells on each side of
        //   a square (or cubic) domain.
        pp.get("n_cell",n_cell);

        // The domain is broken into boxes of size max_grid_size
        pp.get("max_grid_size",max_grid_size);

        // TDGL right hand side parameters
        pp.get("epsilon",epsilon);// epsilon_0*epsilon_r
        pp.get("alpha",alpha);
        pp.get("beta",gamma);
        pp.get("gamma",gamma);
        pp.get("BigGamma",BigGamma);
        pp.get("g11",g11);
        pp.get("g44",g44);

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
    IntVect dom_hi(AMREX_D_DECL(n_cell-1, n_cell-1, n_cell-1));

    // Make a single box that is the entire domain
    Box domain(dom_lo, dom_hi);

    // Initialize the boxarray "ba" from the single box "domain"
    ba.define(domain);

    // Break up boxarray "ba" into chunks no larger than "max_grid_size" along a direction
    ba.maxSize(max_grid_size);

    // This defines the physical box, [0,1] in each direction.
    RealBox real_box({AMREX_D_DECL( prob_lo[0], prob_lo[1], prob_lo[2])},
                     {AMREX_D_DECL( prob_hi[0], prob_hi[1], prob_hi[2])});

    // periodic in all direction
    Array<int,AMREX_SPACEDIM> is_periodic{AMREX_D_DECL(1,1,1)};

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
    MultiFab PoissonRHS(ba, dm, 1, 0);
    MultiFab PoissonPhi(ba, dm, 1, 1);

    //Solver for Poisson equation
    LPInfo info;
    MLABecLaplacian mlabec({geom}, {ba}, {dm}, info);

    // order of stencil
    int linop_maxorder = 2;
    mlabec.setMaxOrder(linop_maxorder);  

    // build array of boundary conditions needed by MLABecLaplacian
    std::array<LinOpBCType, AMREX_SPACEDIM> lo_mlmg_bc;
    std::array<LinOpBCType, AMREX_SPACEDIM> hi_mlmg_bc; 

    //Periodic 
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
    lo_mlmg_bc[idim] = hi_mlmg_bc[idim] = LinOpBCType::Periodic;
    } 

    mlabec.setDomainBC(lo_mlmg_bc,hi_mlmg_bc);

    // coefficients for solver
    MultiFab alpha_cc(ba, dm, 1, 0);
    std::array< MultiFab, AMREX_SPACEDIM > beta_face;
    AMREX_D_TERM(beta_face[0].define(convert(ba,IntVect(1,0,0)), dm, 1, 0);,
                 beta_face[1].define(convert(ba,IntVect(0,1,0)), dm, 1, 0);,
                 beta_face[2].define(convert(ba,IntVect(0,0,1)), dm, 1, 0););
    
    // set cell-centered alpha coefficient to zero
    alpha_cc.setVal(0.);
    
    // set face-centered beta coefficient to epsilon
    AMREX_D_TERM(beta_face[0].setVal(epsilon);,
                 beta_face[1].setVal(epsilon);,
                 beta_face[2].setVal(epsilon););
    
    // set any Dirichlet or Neumann bc's by reading in the ghost cell values
    mlabec.setLevelBC(0, &PoissonPhi);
    
    // (A*alpha_cc - B * div beta grad) phi = rhs
    mlabec.setScalars(0.0, 1.0);
    mlabec.setACoeffs(0, alpha_cc); //First argument 0 is lev
    mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(beta_face));  

    // time = starting time in the simulation
    Real time = 0.0;

    // **********************************
    // INITIALIZE DATA

    // loop over boxes
    for (MFIter mfi(P_old); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& pOld = P_old.array(mfi);

        // set phi = 1 + e^(-(r-0.5)^2)
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
            Real x = (i+0.5) * dx[0];
            Real y = (j+0.5) * dx[1];
            Real z= (k+0.5) * dx[2];
            Real rsquared = ((x-0.5)*(x-0.5)+(y-0.5)*(y-0.5)+(z-0.5)*(z-0.5))/0.01;
            pOld(i,j,k) = 1. + std::exp(-rsquared);
        });
    }

    // Write a plotfile of the initial data if plot_int > 0
    if (plot_int > 0)
    {
        int step = 0;
        const std::string& pltfile = amrex::Concatenate("plt",step,5);
        WriteSingleLevelPlotfile(pltfile, P_old, {"phi"}, geom, time, 0);
    }

    for (int step = 1; step <= nsteps; ++step)
    {
        // fill periodic ghost cells
        P_old.FillBoundary(geom.periodicity());

        // Initialize right hand side 

        for ( MFIter mfi(P_old); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real>& pOld = P_old.array(mfi);
            const Array4<Real>& RHS = PoissonRHS.array(mfi);

            // advance the data by dt
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                RHS(i,j,k) = (pOld(i,j,k+1) - pOld(i,j,k-1))/(2.*dx[2]);
            });
        }

        
 
        //Initial guess for phi
        PoissonPhi.setVal(0.);

        MLMG mlmg(mlabec);
        mlmg.solve({&PoissonPhi}, {&PoissonRHS}, 1.e-10, -1); //1e-10 for rel_tol and -1 (to ignore) 
        //mlmg.solve({&PoissonPhi}, {&PoissonRHS}, mg_rel_tol, mg_abs_tol); //1e-10 for rel_tol and -1 (to ignore) 

        // loop over boxes
        for ( MFIter mfi(P_old); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real>& pOld = P_old.array(mfi);
            const Array4<Real>& pNew = P_new.array(mfi);
            const Array4<Real>& phi = PoissonPhi.array(mfi);


            // advance the data by dt
            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                pNew(i,j,k) = pOld(i,j,k) - dt * BigGamma *
                    (  alpha*pOld(i,j,k) + beta*std::pow(pOld(i,j,k),3.) + gamma*std::pow(pOld(i,j,k),5.)
                     - g44 * (pOld(i+1,j,k) - 2.*pOld(i,j,k) + pOld(i-1,j,k)) / (dx[0]*dx[0])
                     - g44 * (pOld(i,j+1,k) - 2.*pOld(i,j,k) + pOld(i,j-1,k)) / (dx[1]*dx[1])
                     - g11 * (pOld(i,j,k+1) - 2.*pOld(i,j,k) + pOld(i,j,k-1)) / (dx[2]*dx[2])
                     + (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2])
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
            const std::string& pltfile = amrex::Concatenate("plt",step,5);
            WriteSingleLevelPlotfile(pltfile, P_new, {"phi"}, geom, time, step);
        }
    }
}
