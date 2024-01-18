#include "ChargeDensity.H"

// Approximation to the Fermi-Dirac Integral of Order 1/2
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real FD_half(const amrex::Real eta)
{
    amrex::Real nu = std::pow(eta, 4.0) + 50.0 + 33.6 * eta * (1.0 - 0.68 * exp(-0.17 * std::pow((eta + 1.0), 2.0)));
    amrex::Real xi = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu, 3./8.));
    amrex::Real integral = std::pow(exp(-eta) + xi, -1.0);
    return integral;
}


// Compute rho in SC region for given phi
void ComputeRho(MultiFab&      PoissonPhi,
                MultiFab&      rho,
                MultiFab&      e_den,
                MultiFab&      p_den,
		const MultiFab& MaterialMask)
{

    //Define acceptor and donor multifabs for doping and fill them with zero.
    MultiFab acceptor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
    MultiFab donor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
    acceptor_den.setVal(0.);
    donor_den.setVal(0.);

    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev

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
      
                //Following: http://dx.doi.org/10.1063/1.4825209

                amrex::Real Ef = 0.0;
                amrex::Real Eg = bandgap;
                amrex::Real Chi = affinity;
                amrex::Real phi_ref = Chi + 0.5*Eg + 0.5*kb*T*log(Nc/Nv)/q;
                amrex::Real Ec_corr = -q*(phi(i,j,k) - phi_ref) - Chi*q;
                amrex::Real Ev_corr = Ec_corr - q*Eg; 

                //g_A is the acceptor ground state degeneracy factor and is equal to 4 
                //because in most semiconductors each acceptor level can accept one hole of either spin 
                //and the impurity level is doubly degenerate as a result of the two degenerate valence bands 
                //(heavy hole and light hole bands) at the \Gamma point.

                //g_D is the donor ground state degeneracy factor and is equal to 2
                //because a donor level can accept one electron with either spin or can have no electron when filled.

                amrex::Real g_A = 4.0;
                amrex::Real g_D = 2.0;

                //The most common acceptor dopant in bulk Si is boron (B), which has Ea = 44 meV
                //The most common donors in bulk Si are phosphorus (P) and arsenic (As), 
                //which have ionization energies of Ed = 46 meV and 54 meV, respectively.

                amrex::Real Ea = acceptor_ionization_energy; //44.0e-3; 
                amrex::Real Ed = donor_ionization_energy; //46.0e-3; 
                          
                if(use_Fermi_Dirac == 1){
                  //Fermi-Dirac

                  Real eta_n = -(Ec_corr - q*Ef)/(kb*T);
                  Real eta_p = -(q*Ef - Ev_corr)/(kb*T);
                  e_den_arr(i,j,k) = Nc*FD_half(eta_n);
                  hole_den_arr(i,j,k) = Nv*FD_half(eta_p);
         
                  acceptor_den_arr(i,j,k) = acceptor_doping/(1.0 + g_A*exp((-q*Ef + q*Ea + q*phi_ref - q*Chi - q*Eg - q*phi(i,j,k))/(kb*T)));
                  donor_den_arr(i,j,k) = donor_doping/(1.0 + g_D*exp( (q*Ef + q*Ed - q*phi_ref + q*Chi + q*phi(i,j,k)) / (kb*T) ));

                  } else {

                  //Maxwell-Boltzmann
                  e_den_arr(i,j,k) =    Nc*exp( -(Ec_corr - q*Ef) / (kb*T) );
                  hole_den_arr(i,j,k) = Nv*exp( -(q*Ef - Ev_corr) / (kb*T) );
               
                  acceptor_den_arr(i,j,k) = acceptor_doping/(1.0 + g_A*exp((-q*Ef + q*Ea + q*phi_ref - q*Chi - q*Eg - q*phi(i,j,k))/(kb*T)));
                  donor_den_arr(i,j,k) = donor_doping/(1.0 + g_D*exp( (q*Ef + q*Ed - q*phi_ref + q*Chi + q*phi(i,j,k)) / (kb*T) ));

                }

		charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));

             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
 }

