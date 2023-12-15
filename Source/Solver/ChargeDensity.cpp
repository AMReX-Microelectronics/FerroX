#include "ChargeDensity.H"

// Define the function
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real f(const amrex::Real Ef, const amrex::Real phi)
{
    amrex::Real Eg = q*(Ec - Ev); //in J
    amrex::Real p, n;

    if (use_Fermi_Dirac == 1) {

       amrex::Real eta_n = (q*phi + Ef)/(kb*T);
       amrex::Real nu_n = std::pow(eta_n, 4.0) + 50.0 + 33.6 * eta_n * (1 - 0.68 * exp(-0.17 * std::pow((eta_n + 1), 2)));
       amrex::Real xi_n = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_n, 3/8));
       amrex::Real FD_half_n = std::pow(exp(-eta_n) + xi_n, -1.0);

       n = 2.0/sqrt(3.14)*Nc*FD_half_n;

       amrex::Real eta_p = (-q*phi - Eg - Ef)/(kb*T);
       amrex::Real nu_p = std::pow(eta_p, 4.0) + 50.0 + 33.6 * eta_p * (1 - 0.68 * exp(-0.17 * std::pow((eta_p + 1), 2)));
       amrex::Real xi_p = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_p, 3/8));
       amrex::Real FD_half_p = std::pow(exp(-eta_p) + xi_p, -1.0);

       p = 2.0/sqrt(3.14)*Nv*FD_half_p;

    } else {

       n = Nc*exp((Ef + q*phi)/(kb*T));
       p = Nv*exp((-q*phi - Eg - Ef)/(kb*T));

    }

    return p - n + donor_doping - acceptor_doping;
}

// Secant method to find the root of the function
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real secantMethod(const amrex::Real x0, const amrex::Real x1, const int max_iter, const amrex::Real tol, const amrex::Real phi)
{
    amrex::Real x_prev = x0; 
    amrex::Real x_current = x1; 

    for (int iter = 0; iter < max_iter; ++iter)
    {   
        amrex::Real delta_x = (f(x_current, phi) * (x_current - x_prev)) / (f(x_current, phi) - f(x_prev, phi));
        x_prev = x_current;
        x_current -= delta_x;

        if (amrex::Math::abs(delta_x) < tol)
        {
            return x_current; // Convergence reached
        }
    }   

    amrex::Print() << "Secant method did not converge within max_iter." << std::endl;
    return x_current;
}


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

                //amrex::Real root = secantMethod(x0, x1, max_iter, tol);
                amrex::Real Ef = secantMethod(q*Ev, q*Ec, 100, 1.e-6, phi(i,j,k));

                amrex::Real Eg = q*(Ec - Ev); //in J

                if(use_Fermi_Dirac == 1){
                  //Fermi-Dirac

                  Real eta_n = (q*phi(i,j,k) + Ef)/(kb*T);
                  Real nu_n = std::pow(eta_n, 4.0) + 50.0 + 33.6 * eta_n * (1 - 0.68 * exp(-0.17 * std::pow((eta_n + 1), 2)));
                  Real xi_n = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_n, 3/8));
                  Real FD_half_n = std::pow(exp(-eta_n) + xi_n, -1.0);

                  e_den_arr(i,j,k) = 2.0/sqrt(3.14)*Nc*FD_half_n;

                  Real eta_p = (-q*phi(i,j,k) - Eg - Ef)/(kb*T);
                  Real nu_p = std::pow(eta_p, 4.0) + 50.0 + 33.6 * eta_p * (1 - 0.68 * exp(-0.17 * std::pow((eta_p + 1), 2)));
                  Real xi_p = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_p, 3/8));
                  Real FD_half_p = std::pow(exp(-eta_p) + xi_p, -1.0);

                  hole_den_arr(i,j,k) = 2.0/sqrt(3.14)*Nv*FD_half_p;
                } else {

                  //Maxwell-Boltzmann
                  e_den_arr(i,j,k) =    Nc*exp( (Ef + q*phi(i,j,k)) / (kb*T) );
                  hole_den_arr(i,j,k) = Nv*exp( (-q*phi(i,j,k) - Eg - Ef) / (kb*T));
               
                }

		//If in channel, set acceptor doping, else (Source/Drain) set donor doping
                if (mask(i,j,k) == 3.0) {
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

