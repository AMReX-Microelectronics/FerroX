#include "Initialization.H"

// INITIALIZE rho in SC region
void InitializePandRho(Array<MultiFab, AMREX_SPACEDIM> &P_old,
                   MultiFab&   Gamma,
                   MultiFab&   rho,
                   MultiFab&   e_den,
                   MultiFab&   p_den,
                   const       Geometry& geom,
		   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi)
{

    if (prob_type == 1) {  //2D : Initialize uniform P in y direction

       amrex::Print() << "==================================""\n"
                         "P is initialized for a 2D problem." "\n"
                         "==================================""\n" ;

    } else if (prob_type == 2) { // 3D : Initialize random P

       amrex::Print() << "==================================""\n"
                         "P is initialized for a 3D problem." "\n"
                         "==================================""\n" ;

    } else if (prob_type == 3) {

       amrex::Print() << "==================================""\n"
                         "P is initialized for convergence test." "\n"
                         "==================================""\n" ;

    } else {
      amrex::Print() << "Undefine problem type!! Set prob_type in input script." "\n"
                       "prob_type = 1 for 2D problems" "\n"
                       "prob_type = 2 for 3D problems" "\n"
                       "prob_type = 3 for convergence tests." "\n";
      amrex::Abort();
    }

    // loop over boxes
    for (MFIter mfi(rho); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        const Array4<Real> &pOld_x = P_old[0].array(mfi);
        const Array4<Real> &pOld_y = P_old[1].array(mfi);
        const Array4<Real> &pOld_z = P_old[2].array(mfi);
        const Array4<Real>& Gam = Gamma.array(mfi);

        Real pi = 3.141592653589793238;

	Real small = dx[2]*1.e-6;

        // set P
        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];
            if (x <= FE_hi[0] + small && x >= FE_lo[0] - small && y <= FE_hi[1] + small && y >= FE_lo[1] - small && z <= FE_hi[2] + small && z >= FE_lo[2] - small) {
               if (prob_type == 1) {  //2D : Initialize uniform P in y direction

                  double tmp = (i%3 + k%4)/5.;
                  pOld_z(i,j,k) = (-1.0 + 2.0*tmp)*0.002;

               } else if (prob_type == 2) { // 3D : Initialize random P

                 pOld_z(i,j,k) = (-1.0 + 2.0*Random(engine))*0.002;

               } else if (prob_type == 3) { // smooth P for convergence tests

                 pOld_z(i,j,k) = 0.002*exp(-(x*x/(2.0*5.e-9*5.e-9) + y*y/(2.0*5.e-9*5.e-9) + (z-1.5*DE_hi[2])*(z - 1.5*DE_hi[2])/(2.0*2.0e-9*2.0e-9)));

               } else {

                 Abort("Invalid prob_type");

               }

               Gam(i,j,k) = BigGamma;
            } else {
               pOld_z(i,j,k) = 0.0;
               Gam(i,j,k) = 0.0;
            }
            pOld_x(i,j,k) = 0.0;
            pOld_y(i,j,k) = 0.0;
        });
        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev

	MultiFab acceptor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
	MultiFab donor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);


        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
             Real x = prob_lo[0] + (i+0.5) * dx[0];
             Real y = prob_lo[1] + (j+0.5) * dx[1];
             Real z = prob_lo[2] + (k+0.5) * dx[2];

             //SC region
             if (x <= SC_hi[0] + small && x >= SC_lo[0] - small && y <= SC_hi[1] + small && y >= SC_lo[1] - small && z <= SC_hi[2] + small && z >= SC_lo[2] - small) {

                  Real Phi = 0.5*(Ec + Ev); //eV
//                hole_den_arr(i,j,k) = Nv*exp(-(Phi - Ev)*1.602e-19/(kb*T));
//                e_den_arr(i,j,k) = Nc*exp(-(Ec - Phi)*1.602e-19/(kb*T));
//                charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));

                  //Approximate FD integral
                  Real eta_n = q*(Phi - Ec)/(kb*T);
                  Real nu_n = std::pow(eta_n, 4.0) + 50.0 + 33.6 * eta_n * (1 - 0.68 * exp(-0.17 * std::pow((eta_n + 1), 2.0)));
                  Real xi_n = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_n, 3/8));
                  Real FD_half_n = std::pow(exp(-eta_n) + xi_n, -1.0);

                  e_den_arr(i,j,k) = 2.0/sqrt(3.14)*Nc*FD_half_n;

                  Real eta_p = q*(Ev - Phi)/(kb*T);
                  Real nu_p = std::pow(eta_p, 4.0) + 50.0 + 33.6 * eta_p * (1 - 0.68 * exp(-0.17 * std::pow((eta_p + 1), 2.0)));
                  Real xi_p = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_p, 3/8));
                  Real FD_half_p = std::pow(exp(-eta_p) + xi_p, -1.0);

                  hole_den_arr(i,j,k) = 2.0/sqrt(3.14)*Nv*FD_half_p;

		  //If in channel, set acceptor doping, else (Source/Drain) set donor doping
                  if (x <= Channel_hi[0] + small && x >= Channel_lo[0] - small && y <= Channel_hi[1] + small && y >= Channel_lo[1] - small && z <= Channel_hi[2] + small && z >= Channel_lo[2] - small) {
		     acceptor_den_arr(i,j,k) = acceptor_doping; 
	             donor_den_arr(i,j,k) = 0.0;
		  } else { // Source / Drain
		     acceptor_den_arr(i,j,k) = 0.0; 
	             donor_den_arr(i,j,k) = donor_doping;
		  }
                  charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));

             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
    for (int i = 0; i < 3; i++){
      // fill periodic ghost cells
      P_old[i].FillBoundary(geom.periodicity());
    }

 }

