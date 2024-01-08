#include "ChargeDensity.H"

/*
// Define the function
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real f(const amrex::Real Ef, const amrex::Real phi, const amrex::Real mask)
{
    amrex::Real p, n, Na, Nd;

    if (use_Fermi_Dirac == 1) {

       amrex::Real eta_n = (q*phi + Ef - q*Ec)/(kb*T);
       amrex::Real nu_n = std::pow(eta_n, 4.0) + 50.0 + 33.6 * eta_n * (1. - 0.68 * exp(-0.17 * std::pow((eta_n + 1.), 2.)));
       amrex::Real xi_n = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_n, 3./8.));
       amrex::Real FD_half_n = std::pow(exp(-eta_n) + xi_n, -1.0);

       n = Nc*FD_half_n;

       amrex::Real eta_p = (-q*phi + q*Ev  - Ef)/(kb*T);
       amrex::Real nu_p = std::pow(eta_p, 4.0) + 50.0 + 33.6 * eta_p * (1. - 0.68 * exp(-0.17 * std::pow((eta_p + 1.), 2.)));
       amrex::Real xi_p = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_p, 3./8.));
       amrex::Real FD_half_p = std::pow(exp(-eta_p) + xi_p, -1.0);

       p = Nv*FD_half_p;

    } else {

       n = Nc*exp((q*phi + Ef - q*Ec)/(kb*T));
       p = Nv*exp((-q*phi + q*Ev - Ef)/(kb*T));

    }

    if (mask == 3.0) {
       Na = acceptor_doping;
       Nd = 0.0;
    } else { // Source / Drain
       Na = 0.0;
       Nd = donor_doping;
    }
    
    return p - n + Nd - Na;
}

// Secant method to find the root of the function
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real secantMethod(const amrex::Real x0, const amrex::Real x1, const int max_iter, const amrex::Real tol, const amrex::Real phi, const amrex::Real mask)
{
    amrex::Real x_prev = x0; 
    amrex::Real x_current = x1; 

    for (int iter = 0; iter < max_iter; ++iter)
    {   
        amrex::Real delta_x = (f(x_current, phi, mask) * (x_current - x_prev)) / (f(x_current, phi, mask) - f(x_prev, phi, mask));
        x_prev = x_current;
        x_current -= delta_x;
   
        //amrex::Print() << "secant iter = " << iter << ", x_current = " << x_current << "\n";
        if (amrex::Math::abs(delta_x) < tol)
        {
            return x_current; // Convergence reached
        }
    }   

    amrex::Print() << "Secant method did not converge within max_iter." << std::endl;
    return x_current;
}

// Bisection method to find the root of the function
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real bisectionMethod(const amrex::Real a, const amrex::Real b, const int max_iter, const amrex::Real tol, const amrex::Real phi, const amrex::Real mask)
{
    amrex::Real a_local = a;
    amrex::Real b_local = b;

    amrex::Real fa = f(a_local, phi, mask);
    amrex::Real fb = f(b_local, phi, mask);

    if (fa * fb > 0.0)
    {   
        amrex::Print () << "fa = " << fa << ", fb = " << fb << "\n";
        amrex::Print() << "Bisection method requires initial guesses with opposite signs." << std::endl;
        return std::numeric_limits<amrex::Real>::quiet_NaN();
    }   

    amrex::Real c = a_local;
    for (int iter = 0; iter < max_iter; ++iter)
    {   
        c = 0.5 * (a_local + b_local);
        amrex::Real fc = f(c, phi, mask);

        if ((fc == 0.0) || (0.5 * (b_local - a_local) < tol))
        {   
            return c; // Convergence reached
        }   

        if (fa * fc > 0.0)
        {   
            a_local = c;
            fa = fc; 
        }   
        else
        {   
            b_local = c;
            fb = fc; 
        }   
    }   

    amrex::Print() << "Bisection method did not converge within max_iter." << std::endl;
    return c;
}
*/

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

                //amrex::Real Ef = secantMethod(q*Ev, q*Ec, 100, 1.e-6, phi(i,j,k), mask(i,j,k));
                amrex::Real Ef = 0.0;
                amrex::Real Eg = bandgap;
                amrex::Real Chi = affinity;
                amrex::Real phi_ref = Chi + 0.5*Eg + 0.5*kb*T*log(Nc/Nv)/q; //affinity Chi = 4.05 eV 
                amrex::Real Ec_corr = -q*(phi(i,j,k) - phi_ref) - Chi*q;
                amrex::Real Ev_corr = Ec_corr - q*Eg; 

                if(use_Fermi_Dirac == 1){
                  //Fermi-Dirac

                  Real eta_n = -(Ec_corr - q*Ef)/(kb*T);
                  Real nu_n = std::pow(eta_n, 4.0) + 50.0 + 33.6 * eta_n * (1.0 - 0.68 * exp(-0.17 * std::pow((eta_n + 1.0), 2.0)));
                  Real xi_n = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_n, 3./8.));
                  Real FD_half_n = std::pow(exp(-eta_n) + xi_n, -1.0);

                  e_den_arr(i,j,k) = Nc*FD_half_n;

                  Real eta_p = -(q*Ef - Ev_corr)/(kb*T);
                  Real nu_p = std::pow(eta_p, 4.0) + 50.0 + 33.6 * eta_p * (1. - 0.68 * exp(-0.17 * std::pow((eta_p + 1.), 2.)));
                  Real xi_p = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_p, 3./8.));
                  Real FD_half_p = std::pow(exp(-eta_p) + xi_p, -1.0);

                  hole_den_arr(i,j,k) = Nv*FD_half_p;
                } else {

                  //Maxwell-Boltzmann
                  e_den_arr(i,j,k) =    Nc*exp( -(Ec_corr - q*Ef) / (kb*T) );
                  hole_den_arr(i,j,k) = Nv*exp( -(q*Ef - Ev_corr) / (kb*T) );
               
                }

		//If in channel, set acceptor doping, else (Source/Drain) set donor doping
                if (mask(i,j,k) == 3.0) { //p-type assuming complete ionization
                   acceptor_den_arr(i,j,k) = acceptor_doping;
                   donor_den_arr(i,j,k) = 0.0;
                } else { //n-type assuming complete ionization
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

