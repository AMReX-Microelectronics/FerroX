#include "ChargeDensity.H"

// Compute rho in SC region for given phi
void ComputeRho(MultiFab&      PoissonPhi,
                MultiFab&      rho,
                MultiFab&      e_den,
                MultiFab&      p_den,
                const          Geometry& geom,
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi)
{
    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
	Real small = dx[2]*1.e-6;

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev
	MultiFab acceptor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
        MultiFab donor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& phi = PoissonPhi.array(mfi);
	const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);


        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
	     Real x = prob_lo[0] + (i+0.5) * dx[0];
             Real y = prob_lo[1] + (j+0.5) * dx[1];
             Real z = prob_lo[2] + (k+0.5) * dx[2];

             if (x <= SC_hi[0] + small && x >= SC_lo[0] - small && y <= SC_hi[1] + small && y >= SC_lo[1] - small && z <= SC_hi[2] + small && z >= SC_lo[2] - small) {

                    //Maxwell-Boltzmann
//                hole_den_arr(i,j,k) = Nv*exp(-(q*phi(i,j,k) - Ev*1.602e-19)/(kb*T));
//                e_den_arr(i,j,k) = Nc*exp(-(Ec*1.602e-19 - q*phi(i,j,k))/(kb*T));
//                charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));

                    //Fermi-Dirac
                    Real eta_n = q*(phi(i,j,k) - Ec)/(kb*T);
                    Real nu_n = std::pow(eta_n, 4.0) + 50.0 + 33.6 * eta_n * (1 - 0.68 * exp(-0.17 * std::pow((eta_n + 1), 2)));
                    Real xi_n = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_n, 3/8));
                    Real FD_half_n = std::pow(exp(-eta_n) + xi_n, -1.0);

                    e_den_arr(i,j,k) = 2.0/sqrt(3.14)*Nc*FD_half_n;

                    Real eta_p = q*(Ev - phi(i,j,k))/(kb*T);
                    Real nu_p = std::pow(eta_p, 4.0) + 50.0 + 33.6 * eta_p * (1 - 0.68 * exp(-0.17 * std::pow((eta_p + 1), 2)));
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
 }

