#include "ChargeDensity.H"

// Compute rho in SC region for given phi
void ComputeRho(MultiFab&      PoissonPhi,
                MultiFab&      rho,
                MultiFab&      e_den,
                MultiFab&      p_den,
                const          Geometry& geom)
{
    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

       // extract dx from the geometry object
       GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        // Calculate charge density from Phi, FerroX::Nc, FerroX::Nv, Ec, and FerroX::Ev

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& phi = PoissonPhi.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
             Real z = FerroX::prob_lo[2] + (k+0.5) * dx[2];

             if(z <= FerroX::SC_hi){ //SC region

                    //Maxwell-Boltzmann
//                hole_den_arr(i,j,k) = FerroX::Nv*exp(-(FerroX::q*phi(i,j,k) - FerroX::Ev*1.602e-19)/(FerroX::kb*FerroX::T));
//                e_den_arr(i,j,k) = FerroX::Nc*exp(-(Ec*1.602e-19 - FerroX::q*phi(i,j,k))/(FerroX::kb*FerroX::T));
//                charge_den_arr(i,j,k) = FerroX::q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));

                    //Fermi-Dirac
                    Real eta_n = FerroX::q*(phi(i,j,k) - FerroX::Ec)/(FerroX::kb*FerroX::T);
                    Real nu_n = std::pow(eta_n, 4.0) + 50.0 + 33.6 * eta_n * (1 - 0.68 * exp(-0.17 * std::pow((eta_n + 1), 2)));
                    Real xi_n = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_n, 3/8));
                    Real FD_half_n = std::pow(exp(-eta_n) + xi_n, -1.0);

                    e_den_arr(i,j,k) = 2.0/sqrt(3.14)*FerroX::Nc*FD_half_n;

                    Real eta_p = FerroX::q*(FerroX::Ev - phi(i,j,k))/(FerroX::kb*FerroX::T);
                    Real nu_p = std::pow(eta_p, 4.0) + 50.0 + 33.6 * eta_p * (1 - 0.68 * exp(-0.17 * std::pow((eta_p + 1), 2)));
                    Real xi_p = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_p, 3/8));
                    Real FD_half_p = std::pow(exp(-eta_p) + xi_p, -1.0);

                    hole_den_arr(i,j,k) = 2.0/sqrt(3.14)*FerroX::Nv*FD_half_p;

                    charge_den_arr(i,j,k) = FerroX::q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));

             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
 }

