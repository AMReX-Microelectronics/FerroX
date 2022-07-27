#include "Initialization.H"

// INITIALIZE rho in SC region
void InitializePandRho(int prob_type,
                   MultiFab&   P_old,
                   MultiFab&   Gamma,
                   MultiFab&   rho,
                   MultiFab&   e_den,
                   MultiFab&   p_den,
                   Real        SC_lo,
                   Real        SC_hi,
                   Real        DE_lo,
                   Real        DE_hi,
                   Real        BigGamma,
                   Real        q,
                   Real        Ec,
                   Real        Ev,
                   Real        kb,
                   Real        T,
                   Real        Nc,
                   Real        Nv,
                   amrex::GpuArray<amrex::Real, 3> prob_lo,
                   amrex::GpuArray<amrex::Real, 3> prob_hi,
                   const       Geometry& geom)
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

        const Array4<Real>& pOld = P_old.array(mfi);
        const Array4<Real>& Gam = Gamma.array(mfi);

        Real pi = 3.141592653589793238;

        // set P
        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];
            if (z <= DE_hi) {
               pOld(i,j,k) = 0.0;
               Gam(i,j,k) = 0.0;
            } else {
               if (prob_type == 1) {  //2D : Initialize uniform P in y direction

                  double tmp = (i%3 + k%4)/5.;
                  pOld(i,j,k) = (-1.0 + 2.0*tmp)*0.002;

               } else if (prob_type == 2) { // 3D : Initialize random P

                 pOld(i,j,k) = (-1.0 + 2.0*Random(engine))*0.002;

               } else if (prob_type == 3) { // smooth P for convergence tests

                 pOld(i,j,k) = 0.002*exp(-(x*x/(2.0*5.e-9*5.e-9) + y*y/(2.0*5.e-9*5.e-9) + (z-1.5*DE_hi)*(z - 1.5*DE_hi)/(2.0*2.0e-9*2.0e-9)));

               } else {

                 Abort("Invalid prob_type");

               }

               Gam(i,j,k) = BigGamma;
            }
        });
        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
             Real z = prob_lo[2] + (k+0.5) * dx[2];

             if(z <= SC_hi){ //SC region

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
                  charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));

             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
    // fill periodic ghost cells
    P_old.FillBoundary(geom.periodicity());

 }

