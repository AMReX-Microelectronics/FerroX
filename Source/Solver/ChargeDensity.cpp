#include "ChargeDensity.H"
#include "DerivativeAlgorithm.H"

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
    amrex::Print() << "Calculating steady-state carrier distribution." << "\n";

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

                amrex::Real Ea = acceptor_ionization_energy;  
                amrex::Real Ed = donor_ionization_energy; 

                amrex::Real Na, Nd;

                if (mask(i,j,k) == 2.0) {//intrinsic
                   Na = 0.0;
                   Nd = 0.0;
                } else if (mask(i,j,k) == 3.0) { // p-type
                   Na = acceptor_doping;
                   Nd = 0.0;
                } else if (mask(i,j,k) == 4.0) { // n-type
                   Na = 0.0;
                   Nd = donor_doping;
                }
                  
                if(use_Fermi_Dirac == 1){
                  //Fermi-Dirac

                  Real eta_n = -(Ec_corr - q*Ef)/(kb*T);
                  Real eta_p = -(q*Ef - Ev_corr)/(kb*T);
                  e_den_arr(i,j,k) = Nc*FD_half(eta_n);
                  hole_den_arr(i,j,k) = Nv*FD_half(eta_p);
         
                  acceptor_den_arr(i,j,k) = Na/(1.0 + g_A*exp((-q*Ef + q*Ea + q*phi_ref - q*Chi - q*Eg - q*phi(i,j,k))/(kb*T)));
                  donor_den_arr(i,j,k) = Nd/(1.0 + g_D*exp( (q*Ef + q*Ed - q*phi_ref + q*Chi + q*phi(i,j,k)) / (kb*T) ));

                  } else {

                  //Maxwell-Boltzmann
                  e_den_arr(i,j,k) =    Nc*exp( -(Ec_corr - q*Ef) / (kb*T) );
                  hole_den_arr(i,j,k) = Nv*exp( -(q*Ef - Ev_corr) / (kb*T) );
               
                  acceptor_den_arr(i,j,k) = Na/(1.0 + g_A*exp((-q*Ef + q*Ea + q*phi_ref - q*Chi - q*Eg - q*phi(i,j,k))/(kb*T)));
                  donor_den_arr(i,j,k) = Nd/(1.0 + g_D*exp( (q*Ef + q*Ed - q*phi_ref + q*Chi + q*phi(i,j,k)) / (kb*T) ));

                }

		charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));

             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
 }


// --- Bernoulli Function Implementation ---
// This is a crucial part of the Sharfetter-Gummel scheme.
// It's defined as Bern(x) = x / (exp(x) - 1).
// Special care is needed for x close to 0 to avoid division by zero (use Taylor expansion).
AMREX_GPU_HOST_DEVICE AMREX_INLINE
amrex::Real Bern(amrex::Real x)
{
    // Use a small epsilon for robustness around x=0
    if (amrex::Math::abs(x) < 1.0e-6) {
        // Taylor expansion for small x: 1 - x/2 + x^2/12 - x^4/720 + ...
        return 1.0 - x/2.0 + x*x/12.0;
    } else {
        return x / (exp(x) - 1.0);
    }
}

// --- CalculateDriftDiffusionCurrents ---
void CalculateDriftDiffusionCurrents(
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Jn, // OUT: Electron current density components (x,y,z)
    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM>& Jp, // OUT: Hole current density components (x,y,z)
    const amrex::MultiFab& e_den,                      // IN: Electron density
    const amrex::MultiFab& p_den,                      // IN: Hole density
    const amrex::MultiFab& MaterialMask,                      // IN: Hole density
    const amrex::MultiFab& PoissonPhi,                        // IN: Electric potential
    const amrex::Geometry& geom                       // IN: Simulation geometry
)
{

    const amrex::Real kBT_over_q = kb * T / q;
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    for (amrex::MFIter mfi(e_den, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();

        // Get Array4 views for electron density, hole density, potential
        amrex::Array4<amrex::Real const> const& e_den_arr = e_den.const_array(mfi);
        amrex::Array4<amrex::Real const> const& p_den_arr = p_den.const_array(mfi);
        amrex::Array4<amrex::Real const> const& phi_arr = PoissonPhi.const_array(mfi);
	amrex::Array4<Real const> const& mask = MaterialMask.const_array(mfi);

        // Get Array4 views for current components (output)
        amrex::Array4<amrex::Real> const& Jnx_arr = Jn[0].array(mfi);
        amrex::Array4<amrex::Real> const& Jny_arr = Jn[1].array(mfi);
        amrex::Array4<amrex::Real> const& Jnz_arr = Jn[2].array(mfi);
        amrex::Array4<amrex::Real> const& Jpx_arr = Jp[0].array(mfi);
        amrex::Array4<amrex::Real> const& Jpy_arr = Jp[1].array(mfi);
        amrex::Array4<amrex::Real> const& Jpz_arr = Jp[2].array(mfi);

        amrex::Real mu_n = electron_mobility;
        amrex::Real mu_p = hole_mobility;

        amrex::Real D_n = electron_diffusion_coefficient;
        amrex::Real D_p = hole_diffusion_coefficient;

        amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) noexcept
        {
            // Initialize current components for this cell (will be overwritten)
            Jnx_arr(i, j, k) = 0.0;
            Jny_arr(i, j, k) = 0.0;
            Jnz_arr(i, j, k) = 0.0;
            Jpx_arr(i, j, k) = 0.0;
            Jpy_arr(i, j, k) = 0.0;
            Jpz_arr(i, j, k) = 0.0;

            if (mask(i,j,k) >= 2.0) {
               // --- Calculate J_x (current across faces normal to x-axis) ---
               // For a cell (i,j,k), we're interested in the current *through* its faces.
               // Jnx_arr(i,j,k) will store the current through the FACE at (i+1/2, j, k).
               // Jpx_arr(i,j,k) will store the current through the FACE at (i+1/2, j, k).
               // Similarly for y and z.

               // Only compute if we are not at the very right boundary of the box for the current's central difference.
               // If the box is not periodic, the currents at the domain boundaries will be handled by specific boundary conditions
               // or implicitly be zero if not computed beyond the physical domain.
               if (i < bx.bigEnd(0)) { // This calculates J_x at (i+1/2, j, k)
                   amrex::Real dPhi = phi_arr(i+1, j, k) - phi_arr(i, j, k);
                   amrex::Real arg = dPhi / kBT_over_q;
                   Jnx_arr(i, j, k) = q * D_n / dx[0] * (e_den_arr(i+1,j,k) * Bern(arg) - e_den_arr(i,j,k) * Bern(-arg));
                   Jpx_arr(i, j, k) = q * D_p / dx[0] * (p_den_arr(i,j,k) * Bern(-arg) - p_den_arr(i+1,j,k) * Bern(arg));
               }

               // --- Calculate J_y (current across faces normal to y-axis) ---
               if (j < bx.bigEnd(1)) { // This calculates J_y at (i, j+1/2, k)
                   amrex::Real dPhi = phi_arr(i, j+1, k) - phi_arr(i, j, k);
                   amrex::Real arg = dPhi / kBT_over_q;
                   Jny_arr(i, j, k) = q * D_n / dx[1] * (e_den_arr(i,j+1,k) * Bern(arg) - e_den_arr(i,j,k) * Bern(-arg));
                   Jpy_arr(i, j, k) = q * D_p / dx[1] * (p_den_arr(i,j,k) * Bern(-arg) - p_den_arr(i,j+1,k) * Bern(arg));
               }

               // --- Calculate J_z (current across faces normal to z-axis) ---
               if (k < bx.bigEnd(2)) { // This calculates J_z at (i, j, k+1/2)
                   amrex::Real dPhi = phi_arr(i, j, k+1) - phi_arr(i, j, k);
                   amrex::Real arg = dPhi / kBT_over_q;
                   Jnz_arr(i, j, k) = q * D_n / dx[2] * (e_den_arr(i,j,k+1) * Bern(arg) - e_den_arr(i,j,k) * Bern(-arg));
                   Jpz_arr(i, j, k) = q * D_p / dx[2] * (p_den_arr(i,j,k) * Bern(-arg) - p_den_arr(i,j,k+1) * Bern(arg));
               }
	    }
        });
    }

    for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        Jn[d].FillBoundary(geom.periodicity());
        Jp[d].FillBoundary(geom.periodicity());
    }
}

// Compute rho in SC region for given phi
void ComputeRho_DriftDiffusion(MultiFab&      PoissonPhi,
                MultiFab&      rho,
                Array<MultiFab, AMREX_SPACEDIM> &Jn,
                Array<MultiFab, AMREX_SPACEDIM> &Jp,
                MultiFab&      e_den,
                MultiFab&      p_den,
                MultiFab&      e_den_old,
                MultiFab&      p_den_old,
                MultiFab& MaterialMask,
                const Geometry& geom)
{

    amrex::Print() << "Calculating carrier transport using Drift-Diffusion model." << "\n";

    //Define acceptor and donor multifabs for doping and fill them with zero.
    MultiFab acceptor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
    MultiFab donor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
    acceptor_den.setVal(0.);
    donor_den.setVal(0.);

    Compute_Current(PoissonPhi, e_den, p_den, Jn, Jp, MaterialMask, geom);

    // First, calculate the current components and store them in Jn and Jp
    CalculateDriftDiffusionCurrents(Jn, Jp, e_den, p_den, MaterialMask, PoissonPhi, geom);

    // Get cell spacing from geometry
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = geom.CellSizeArray();

    // Loop over grids (boxes) in the MultiFab for updating densities
    for (amrex::MFIter mfi(e_den, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.tilebox();
        
        // Get Array4 views for densities (for update)
        amrex::Array4<amrex::Real> const& e_den_arr = e_den.array(mfi);
        amrex::Array4<amrex::Real> const& p_den_arr = p_den.array(mfi);
        amrex::Array4<amrex::Real> const& charge_den_arr = rho.array(mfi);
        amrex::Array4<amrex::Real> const& phi = PoissonPhi.array(mfi);
	const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);
        const Array4<Real>& mask = MaterialMask.array(mfi);

        // Get Array4 views for current components (from `CalculateDriftDiffusionCurrents`)
        amrex::Array4<amrex::Real const> const& Jnx_arr = Jn[0].const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jny_arr = Jn[1].const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jnz_arr = Jn[2].const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jpx_arr = Jp[0].const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jpy_arr = Jp[1].const_array(mfi);
        amrex::Array4<amrex::Real const> const& Jpz_arr = Jp[2].const_array(mfi);
        
        amrex::ParallelFor(bx, [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) noexcept
        {
	
	    amrex::Real Ef = 0.0;
            amrex::Real Eg = bandgap;
            amrex::Real Chi = affinity;
            amrex::Real phi_ref = Chi + 0.5*Eg + 0.5*kb*T*log(Nc/Nv)/q;

             if (mask(i,j,k) >= 2.0) {
                
	        // --- Calculate Divergence ---
                // div(J) = (J_x_R - J_x_L)/dx + (J_y_R - J_y_L)/dy + (J_z_R - J_z_L)/dz
                // Note: J_x_R for cell (i,j,k) is Jnx_arr(i,j,k) (current at i+1/2 face)
                //       J_x_L for cell (i,j,k) is Jnx_arr(i-1,j,k) (current at i-1/2 face)

                amrex::Real div_Jn = (Jnx_arr(i, j, k) - Jnx_arr(i-1, j, k)) / dx[0] +
                                     (Jny_arr(i, j, k) - Jny_arr(i, j-1, k)) / dx[1] +
                                     (Jnz_arr(i, j, k) - Jnz_arr(i, j, k-1)) / dx[2];

                amrex::Real div_Jp = (Jpx_arr(i, j, k) - Jpx_arr(i-1, j, k)) / dx[0] +
                                     (Jpy_arr(i, j, k) - Jpy_arr(i, j-1, k)) / dx[1] +
                                     (Jpz_arr(i, j, k) - Jpz_arr(i, j, k-1)) / dx[2];

                // --- Update Densities (ignoring recombination) ---
                e_den_arr(i, j, k) += dt * (1.0/q * div_Jn);
                p_den_arr(i, j, k) += dt * (-1.0/q * div_Jp);
      
                //g_A is the acceptor ground state degeneracy factor and is equal to 4 
                //because in most semiconductors each acceptor level can accept one hole of either spin 
                //and the impurity level is doubly degenerate as a result of the two degenerate valence bands 
                //(heavy hole and light hole bands) at the \Gamma point.

                //g_D is the donor ground state degeneracy factor and is equal to 2
                //because a donor level can accept one electron with either spin or can have no electron when filled.

                amrex::Real g_A = 4.0;
                amrex::Real g_D = 2.0;

                amrex::Real Ea = acceptor_ionization_energy;  
                amrex::Real Ed = donor_ionization_energy; 

                amrex::Real Na, Nd;

                if (mask(i,j,k) == 2.0) {//intrinsic
                   Na = 0.0;
                   Nd = 0.0;
                } else if (mask(i,j,k) == 3.0) { // p-type
                   Na = acceptor_doping;
                   Nd = 0.0;
                } else if (mask(i,j,k) == 4.0) { // n-type
                   Na = 0.0;
                   Nd = donor_doping;
                }

                acceptor_den_arr(i,j,k) = Na/(1.0 + g_A*exp((-q*Ef + q*Ea + q*phi_ref - q*Chi - q*Eg - q*phi(i,j,k))/(kb*T)));
                donor_den_arr(i,j,k) = Nd/(1.0 + g_D*exp( (q*Ef + q*Ed - q*phi_ref + q*Chi + q*phi(i,j,k)) / (kb*T) ));

	        charge_den_arr(i,j,k) = q*(p_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));
	     }
        });
    }

    // After updating, fill ghost cells for density multifabs if needed
    e_den.FillBoundary(geom.periodicity());
    p_den.FillBoundary(geom.periodicity());
    rho.FillBoundary(geom.periodicity());
    /*
    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev

        const Array4<Real>& hole_den_arr_old = p_den_old.array(mfi);
        const Array4<Real>& e_den_arr_old = e_den_old.array(mfi);
        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& phi = PoissonPhi.array(mfi);
	const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);
        const Array4<Real>& mask = MaterialMask.array(mfi);

        const Array4<Real>& Jnx_arr = Jn[0].array(mfi);
        const Array4<Real>& Jny_arr = Jn[1].array(mfi);
        const Array4<Real>& Jnz_arr = Jn[2].array(mfi);

        const Array4<Real>& Jpx_arr = Jp[0].array(mfi);
        const Array4<Real>& Jpy_arr = Jp[1].array(mfi);
        const Array4<Real>& Jpz_arr = Jp[2].array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

             amrex::Real Ef = 0.0;
             amrex::Real Eg = bandgap;
             amrex::Real Chi = affinity;
             amrex::Real phi_ref = Chi + 0.5*Eg + 0.5*kb*T*log(Nc/Nv)/q;

             if (mask(i,j,k) >= 2.0) {
      
                //g_A is the acceptor ground state degeneracy factor and is equal to 4 
                //because in most semiconductors each acceptor level can accept one hole of either spin 
                //and the impurity level is doubly degenerate as a result of the two degenerate valence bands 
                //(heavy hole and light hole bands) at the \Gamma point.

                //g_D is the donor ground state degeneracy factor and is equal to 2
                //because a donor level can accept one electron with either spin or can have no electron when filled.

                amrex::Real g_A = 4.0;
                amrex::Real g_D = 2.0;

                amrex::Real Ea = acceptor_ionization_energy;  
                amrex::Real Ed = donor_ionization_energy; 

                amrex::Real Na, Nd;

                if (mask(i,j,k) == 2.0) {//intrinsic
                   Na = 0.0;
                   Nd = 0.0;
                } else if (mask(i,j,k) == 3.0) { // p-type
                   Na = acceptor_doping;
                   Nd = 0.0;
                } else if (mask(i,j,k) == 4.0) { // n-type
                   Na = 0.0;
                   Nd = donor_doping;
                }
                  
                e_den_arr(i,j,k) += 1./q*dt*( DFDx(Jnx_arr, i,j,k,dx) + DFDy(Jny_arr, i,j,k,dx) + DFDz(Jnz_arr, mask,i,j,k,dx) );
                hole_den_arr(i,j,k) += - 1./q*dt*( DFDx(Jpx_arr, i,j,k,dx) + DFDy(Jpy_arr, i,j,k,dx) + DFDz(Jpz_arr, mask, i,j,k,dx) );
         
                acceptor_den_arr(i,j,k) = Na/(1.0 + g_A*exp((-q*Ef + q*Ea + q*phi_ref - q*Chi - q*Eg - q*phi(i,j,k))/(kb*T)));
                donor_den_arr(i,j,k) = Nd/(1.0 + g_D*exp( (q*Ef + q*Ed - q*phi_ref + q*Chi + q*phi(i,j,k)) / (kb*T) ));

		charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));

             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
    e_den.FillBoundary(geom.periodicity());
    p_den.FillBoundary(geom.periodicity());
    rho.FillBoundary(geom.periodicity());
    */
 }

void Compute_Current(MultiFab&      PoissonPhi,
                     MultiFab&      e_den,
                     MultiFab&      p_den, 
                     Array<MultiFab, AMREX_SPACEDIM> &Jn,
                     Array<MultiFab, AMREX_SPACEDIM> &Jp,
                     MultiFab& MaterialMask,
                     const Geometry& geom)
{
 
    MultiFab e_potential(PoissonPhi.boxArray(), PoissonPhi.DistributionMap(), 1, 1);
    MultiFab p_potential(PoissonPhi.boxArray(), PoissonPhi.DistributionMap(), 1, 1);
    e_potential.setVal(0.);
    p_potential.setVal(0.);

    Compute_Effective_Potentials(PoissonPhi, e_potential, p_potential, geom);

    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        //const Box& bx = mfi.growntilebox(1);
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& phi = PoissonPhi.array(mfi);
        const Array4<Real>& e_phi_arr = e_potential.array(mfi);
        const Array4<Real>& p_phi_arr = p_potential.array(mfi);
        const Array4<Real>& mask = MaterialMask.array(mfi);

        const Array4<Real>& Jnx_arr = Jn[0].array(mfi);
        const Array4<Real>& Jny_arr = Jn[1].array(mfi);
        const Array4<Real>& Jnz_arr = Jn[2].array(mfi);

        const Array4<Real>& Jpx_arr = Jp[0].array(mfi);
        const Array4<Real>& Jpy_arr = Jp[1].array(mfi);
        const Array4<Real>& Jpz_arr = Jp[2].array(mfi);

        amrex::Real mu_n = electron_mobility;
        amrex::Real mu_p = hole_mobility;

        amrex::Real D_n = electron_diffusion_coefficient;
        amrex::Real D_p = hole_diffusion_coefficient;

        //amrex::Print()<< "mu_n = " << mu_n << "\n";
        //amrex::Print()<< "mu_p = " << mu_p << "\n";
        //
        //amrex::Print()<< "D_n = " << D_n << "\n";
        //amrex::Print()<< "D_p = " << D_p << "\n";
        
        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

           if (mask(i,j,k) >= 2.0) {
              Real Fn_x = DFDx(e_phi_arr, i, j, k, dx);
              Real Fn_y = DFDy(e_phi_arr, i, j, k, dx);
              Real Fn_z = DFDz(e_phi_arr, mask, i, j, k, dx); 

              //Real Fn_x = DFDx(phi, i, j, k, dx);
              //Real Fn_y = DFDy(phi, i, j, k, dx);
              //Real Fn_z = DFDz(phi, mask, i, j, k, dx); 

              Real gradn_x = DFDx(e_den_arr, i, j, k, dx);
              Real gradn_y = DFDy(e_den_arr, i, j, k, dx);
              Real gradn_z = DFDz(e_den_arr, mask, i, j, k, dx); 

              Real Fp_x = DFDx(p_phi_arr, i, j, k, dx);
              Real Fp_y = DFDy(p_phi_arr, i, j, k, dx);
              Real Fp_z = DFDz(p_phi_arr, mask, i, j, k, dx); 

              //Real Fp_x = DFDx(phi, i, j, k, dx);
              //Real Fp_y = DFDy(phi, i, j, k, dx);
              //Real Fp_z = DFDz(phi, mask, i, j, k, dx); 

              Real gradp_x = DFDx(hole_den_arr, i, j, k, dx);
              Real gradp_y = DFDy(hole_den_arr, i, j, k, dx);
              Real gradp_z = DFDz(hole_den_arr, mask, i, j, k, dx);

              Jnx_arr(i,j,k) = e_den_arr(i,j,k)*mu_n*Fn_x + q*D_n*gradn_x;           
              Jny_arr(i,j,k) = e_den_arr(i,j,k)*mu_n*Fn_y + q*D_n*gradn_y;           
              Jnz_arr(i,j,k) = e_den_arr(i,j,k)*mu_n*Fn_z + q*D_n*gradn_z;           

              Jpx_arr(i,j,k) = hole_den_arr(i,j,k)*mu_p*Fp_x - q*D_p*gradp_x;           
              Jpy_arr(i,j,k) = hole_den_arr(i,j,k)*mu_p*Fp_y - q*D_p*gradp_y;           
              Jpz_arr(i,j,k) = hole_den_arr(i,j,k)*mu_p*Fp_z - q*D_p*gradp_z;

              //if(Fn_x != 0.0 || Fn_y != 0.0 || Fn_z != 0.0 || Fp_x != 0.0 || Fp_y != 0.0 || Fp_z != 0.0){
	      //   
	      //   amrex::Print() << "Fn_x(" << i << "," << j << ","<< k << ") = " << Fn_x <<  "\n";
	      //   amrex::Print() << "Fn_y(" << i << "," << j << ","<< k << ") = " << Fn_y <<  "\n";
	      //   amrex::Print() << "Fn_z(" << i << "," << j << ","<< k << ") = " << Fn_z <<  "\n";
	      //   amrex::Print() << "Fp_x(" << i << "," << j << ","<< k << ") = " << Fp_x <<  "\n";
	      //   amrex::Print() << "Fp_y(" << i << "," << j << ","<< k << ") = " << Fp_y <<  "\n";
	      //   amrex::Print() << "Fp_z(" << i << "," << j << ","<< k << ") = " << Fp_z <<  "\n";
	      //
	      //}	 
	      if(Fn_x != 0.0 && Jnx_arr(i,j,k) == 0.0) {
		 amrex::Print() << "Fn_x(" << i << "," << j << ","<< k << ") = " << Fn_x <<", Jnx_arr(i,j,k) = " << Jnx_arr(i,j,k) <<  "\n";
	      }

	      //if(i == 16 && j == 16 && k == 26){ 
	      //   amrex::Print() << "Jnx_arr(" << i << "," << j << ","<< k << ") = " << Jnx_arr(i,j,k) << "gradn_x = " << gradn_x << "\n";
	      //   amrex::Print() << "Jny_arr(" << i << "," << j << ","<< k << ") = " << Jny_arr(i,j,k) << "gradn_x = " << gradn_y << "\n";
	      //   amrex::Print() << "Jnz_arr(" << i << "," << j << ","<< k << ") = " << Jnz_arr(i,j,k) << "gradn_x = " << gradn_z << "\n";
	      //   amrex::Print() << "Jpx_arr(" << i << "," << j << ","<< k << ") = " << Jpx_arr(i,j,k) << "gradn_x = " << gradp_x << "\n";
	      //   amrex::Print() << "Jpy_arr(" << i << "," << j << ","<< k << ") = " << Jpy_arr(i,j,k) << "gradn_x = " << gradp_y << "\n";
	      //   amrex::Print() << "Jpz_arr(" << i << "," << j << ","<< k << ") = " << Jpz_arr(i,j,k) << "gradn_x = " << gradp_z << "\n";
	      //}
           } else {
              Jnx_arr(i,j,k) = 0.0;
              Jny_arr(i,j,k) = 0.0;
              Jnz_arr(i,j,k) = 0.0;

              Jpx_arr(i,j,k) = 0.0;
              Jpy_arr(i,j,k) = 0.0;
              Jpz_arr(i,j,k) = 0.0;
           }    

   	});
    }

    for (int i = 0; i < 3; i++){
        Jn[i].FillBoundary(geom.periodicity());
        Jp[i].FillBoundary(geom.periodicity());
    } 
}


void Compute_Effective_Potentials(const MultiFab& PoissonPhi,
                                        MultiFab& e_potential,
                                        MultiFab& p_potential,
                                        const Geometry& geom)
{

// loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();
        //const Box& bx = mfi.growntilebox(1);

        const Array4<Real const>& phi = PoissonPhi.array(mfi);
        const Array4<Real>& e_phi = e_potential.array(mfi);
        const Array4<Real>& p_phi = p_potential.array(mfi);

        amrex::Real Ef = 0.0;
        amrex::Real Eg = bandgap;
        amrex::Real Chi = affinity;
        amrex::Real phi_ref = Chi + 0.5*Eg + 0.5*kb*T*log(Nc/Nv)/q;

        amrex::Real Delta_Eg = 0.; //50.0*1e-3*q; //band_gap_narrowing; ??TODO: use slotboom approxmiation. Delta_Eg = 0.0 is valid for non-degenerate SC, and can be used with Maxwell-Boltzmann formulation

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

            amrex::Real Ec = -q*(phi(i,j,k) - phi_ref) - Chi*q;
            amrex::Real Ev = Ec - q*Eg; 

            amrex::Real eta_n = -(Ec - q*Ef)/(kb*T);
            amrex::Real eta_p = -(q*Ef - Ev)/(kb*T);
            amrex::Real gamma_n = 1.; //FD_half(eta_n)/exp(eta_n); //gamma_n = 1. = gamma_p reduces to Maxwell-Boltzmann formulation and should work for non-degenerate SC
            amrex::Real gamma_p = 1.; //FD_half(eta_p)/exp(eta_p);
            amrex::Real E_i = q*phi_ref - q*Chi - q*phi(i,j,k) - 0.5*q*Eg - 0.5*kb*T*log( (Nc*gamma_n) / (Nv*gamma_p) );
            e_phi(i,j,k) = E_i - 0.5*Delta_Eg - 0.5*kb*T*log(gamma_n*gamma_p); 
            p_phi(i,j,k) = E_i + 0.5*Delta_Eg + 0.5*kb*T*log(gamma_n*gamma_p);

        });
    }

    e_potential.FillBoundary(geom.periodicity());
    p_potential.FillBoundary(geom.periodicity());
}
