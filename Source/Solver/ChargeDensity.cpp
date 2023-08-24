#include "ChargeDensity.H"

// Compute rho in SC region for given phi
void ComputeRho(MultiFab&      PoissonPhi,
                MultiFab&      rho,
                MultiFab&      e_den,
                MultiFab&      p_den,
		const MultiFab& MaterialMask)
{
    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev
	MultiFab acceptor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
        MultiFab donor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& phi = PoissonPhi.array(mfi);
	const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);
        const Array4<Real const>& mask = MaterialMask.array(mfi);


        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

             if (mask(i,j,k) >= 2.0) {

                if(use_Fermi_Dirac == 1){
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
                } else {
                  //Maxwell-Boltzmann
                  acceptor_den_arr(i,j,k) = acceptor_doping;
                  donor_den_arr(i,j,k) = 0.0;
                  Real p_0 = acceptor_doping;
                  Real n_0 = intrinsic_carrier_concentration*intrinsic_carrier_concentration/p_0;
                  hole_den_arr(i,j,k) = p_0*exp(-(q*phi(i,j,k))/(kb*T));
                  e_den_arr(i,j,k) =    n_0*exp(q*phi(i,j,k)/(kb*T));
                }

		////If in channel, set acceptor doping, else (Source/Drain) set donor doping
                //if (mask(i,j,k) == 3.0) {
                //   acceptor_den_arr(i,j,k) = acceptor_doping;
                //   donor_den_arr(i,j,k) = 0.0;
                //} else { // Source / Drain
                //   acceptor_den_arr(i,j,k) = 0.0;
                //   donor_den_arr(i,j,k) = donor_doping;
                //}

		charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));

             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
 }
