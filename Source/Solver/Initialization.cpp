#include "Initialization.H"
#include "Utils/eXstaticUtils/eXstaticUtil.H"
#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"
#define MATERIAL_UNKNOWN -1.0 // A unique value for undefined/overlapping regions
			      
// INITIALIZE rho in SC region
void InitializePandRho(Array<MultiFab, AMREX_SPACEDIM> &P_old,
                   MultiFab&   Gamma,
                   MultiFab&   rho,
                   MultiFab&   e_den,
                   MultiFab&   p_den,
                   MultiFab&   acceptor_den,
                   MultiFab&   donor_den,
		   const MultiFab& MaterialMask,
		   const MultiFab& tphaseMask,
                   const amrex::GpuArray<int, AMREX_SPACEDIM>& n_cell,
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

    // Read this from inputs file. Default seed = 1
    int seed = random_seed;

    int nprocs = ParallelDescriptor::NProcs();

    if (prob_type == 1) {
       amrex::InitRandom(seed                             , nprocs, seed                             );  // give all MPI ranks the same seed
    } else { 
      amrex::InitRandom(seed+ParallelDescriptor::MyProc(), nprocs, seed+ParallelDescriptor::MyProc());  // give all MPI ranks a different seed
    }

    int nrand = n_cell[0]*n_cell[2];
    amrex::Gpu::ManagedVector<Real> rngs(nrand, 0.0);

    // generate random numbers on the host
    for (int i=0; i<nrand; ++i) {
        //rngs[i] = amrex::RandomNormal(0.,1.); // zero mean, unit variance
         rngs[i] = amrex::Random(); // uniform [0,1] option
    }

    // loop over boxes
    for (MFIter mfi(rho); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        const Array4<Real> &pOld_p = P_old[0].array(mfi);
        const Array4<Real> &pOld_q = P_old[1].array(mfi);
        const Array4<Real> &pOld_r = P_old[2].array(mfi);
        const Array4<Real>& Gam = Gamma.array(mfi);
        const Array4<Real const>& mask = MaterialMask.array(mfi);
        const Array4<Real const>& tphase = tphaseMask.array(mfi);

        Real* rng = rngs.data();

        // set P
        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];
            if (mask(i,j,k) == 0.0) { //FE mask is 0.0
               if (prob_type == 1) {  //2D : Initialize uniform P in y direction

                 pOld_p(i,j,k) = (-1.0 + 2.0*rng[i + k*n_cell[2]])*Remnant_P[0];
                 pOld_q(i,j,k) = (-1.0 + 2.0*rng[i + k*n_cell[2]])*Remnant_P[1];
                 pOld_r(i,j,k) = (-1.0 + 2.0*rng[i + k*n_cell[2]])*Remnant_P[2];

               } else if (prob_type == 2) { // 3D : Initialize random P

                 pOld_p(i,j,k) = (-1.0 + 2.0*Random(engine))*Remnant_P[0];
                 pOld_q(i,j,k) = (-1.0 + 2.0*Random(engine))*Remnant_P[1];
                 pOld_r(i,j,k) = (-1.0 + 2.0*Random(engine))*Remnant_P[2];

               } else if (prob_type == 3) { // smooth P for convergence tests

                 pOld_p(i,j,k) = Remnant_P[0]*exp(-(x*x/(2.0*5.e-9*5.e-9) + y*y/(2.0*5.e-9*5.e-9) + (z-1.5*DE_hi[2])*(z - 1.5*DE_hi[2])/(2.0*2.0e-9*2.0e-9)));
                 pOld_q(i,j,k) = Remnant_P[1]*exp(-(x*x/(2.0*5.e-9*5.e-9) + y*y/(2.0*5.e-9*5.e-9) + (z-1.5*DE_hi[2])*(z - 1.5*DE_hi[2])/(2.0*2.0e-9*2.0e-9)));
                 pOld_r(i,j,k) = Remnant_P[2]*exp(-(x*x/(2.0*5.e-9*5.e-9) + y*y/(2.0*5.e-9*5.e-9) + (z-1.5*DE_hi[2])*(z - 1.5*DE_hi[2])/(2.0*2.0e-9*2.0e-9)));

               } else {

                 Abort("Invalid prob_type");

               }

               Gam(i,j,k) = BigGamma;

	       //set t_phase Pz to zero
	       //if(x <= t_phase_hi[0] && x >= t_phase_lo[0] && y <= t_phase_hi[1] && y >= t_phase_lo[1] && z <= t_phase_hi[2] && z >= t_phase_lo[2]){
	       if(tphase(i,j,k) == 1.0){
                 pOld_r(i,j,k) = 0.0;
	       }

            } else {
               pOld_p(i,j,k) = 0.0;
               pOld_q(i,j,k) = 0.0;
               pOld_r(i,j,k) = 0.0;
               Gam(i,j,k) = 0.0;
            }

	    if (is_polarization_scalar == 1){
               pOld_p(i,j,k) = 0.0;
               pOld_q(i,j,k) = 0.0;
	    }
        });
    }

    for (int i = 0; i < 3; i++){
      // fill periodic ghost cells
      P_old[i].FillBoundary(geom.periodicity());
    }
	
    // Calculate charge density from Phi, Nc, Nv, Ec, and Ev

    // loop over boxes
    for (MFIter mfi(rho); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        //const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);
        const Array4<Real const>& mask = MaterialMask.array(mfi);

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
	/*
        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real Na_val, Nd_val; // Use temporary values for N_A and N_D for this cell
            amrex::Real initial_n, initial_p;

            Real cell_z = prob_lo[2] + (k+0.5) * dx[2];

            const amrex::Real z_junction = 0.5e-6; // Center of the p-n junction
            const amrex::Real junction_width = 0.05e-6; // Characteristic width of the transition (adjust as needed)

            // SC region (mask >= 2.0 indicates semiconductor)
            if (mask(i,j,k) >= 2.0) {

                // Smooth doping profile for p-n junction
                Nd_val = 0.5 * donor_doping * (1.0 - std::tanh((cell_z - z_junction) / junction_width));
                Na_val = 0.5 * acceptor_doping * (1.0 + std::tanh((cell_z - z_junction) / junction_width));

                // If explicitly an intrinsic region, override doping (optional, depends on model)
                if (mask(i,j,k) == 2.0) {
                    Na_val = 0.0;
                    Nd_val = 0.0;
                }

                // --- MODIFIED INITIALIZATION FOR CARRIER CONCENTRATIONS ---
                amrex::Real ni = intrinsic_carrier_concentration;
                amrex::Real net_doping = Nd_val - Na_val;

                // Simple approach: Assume majority carriers are equal to net doping,
                // minority carriers follow mass action. This works for neutral regions.
                // For the depletion region, we will use a more explicit model.

                // This is a simplified depletion approximation for initial guess.
                // It's still approximate and the solver will refine it.
                if (net_doping > 0.0) { // Effectively n-type
                    initial_n = std::max(Nd_val - Na_val, ni); // Assume n ~ Nd - Na, but not less than ni
                    initial_p = ni * ni / initial_n;
                } else if (net_doping < 0.0) { // Effectively p-type
                    initial_p = std::max(Na_val - Nd_val, ni); // Assume p ~ Na - Nd, but not less than ni
                    initial_n = ni * ni / initial_p;
                } else { // Intrinsic or perfectly compensated
                    initial_n = ni;
                    initial_p = ni;
                }

                // Now, the crucial part: if we are in the *anticipated* depletion region,
                // we might want to override carrier concentrations to reflect charge.
                // This is still an approximation for initial guess.
                // A better way is to solve Poisson with initial potential guess.

                // Let's refine the initial_n and initial_p calculation to explicitly consider the
                // charge density directly in the formula, rather than assuming local neutrality.
                // This is typically done by solving the full drift-diffusion-Poisson system.
                // For a *pure initialization of charge for Poisson*, you can often simplify.

                // Let's go back to the idea that in neutral regions, the charge is zero,
                // and in the depletion region, it's roughly the ionized dopants.
                // The tanh function *already* defines the doping profile.
                // The most robust way is to provide a *potential* profile guess,
                // then derive n and p from it, and then calculate charge.

                // For a good initial guess for Poisson, you often calculate:
                // rho = q * (p - n + Nd+ - Na-)
                // and then you solve Poisson for phi.
                //
                // The issue you're facing is that your *initial guess for n and p* already
                // enforces p - n - Na + Nd = 0, which makes rho vanish.
                //
                // If you want to *see* charge at initialization, you must provide an initial
                // guess for n and p that *does not* locally enforce neutrality in the depletion region.

                // REVISED INITIALIZATION FOR CHARGE_DEN_ARR:
                // Let's try to directly define the charge density for a simplified
                // depletion region, primarily driven by the dopants.

                // Assign initial carrier concentrations to MultiFabs
                // These are just *guesses* to kick off the solver.
                // The solver will find the true equilibrium n and p.
                hole_den_arr(i,j,k) = initial_p; // Based on local neutrality in bulk
                e_den_arr(i,j,k) = initial_n;   // Based on local neutrality in bulk

                // Assign doping concentrations to MultiFabs
                acceptor_den_arr(i,j,k) = Na_val;
                donor_den_arr(i,j,k) = Nd_val;

                // Calculate the charge density for Poisson's RHS
                // THIS IS THE CRITICAL PART FOR SEEING INITIAL CHARGE
                // The charge density *is* primarily due to the fixed dopants in depletion.
                // Mobile carriers will be very low in the depletion region.

                // For an initial guess, one common way is to assume full ionization of dopants
                // and negligible mobile carriers *in the regions that will become depleted*.
                // However, defining these regions precisely without a potential guess is tricky.

                // Let's stick with the definition: charge_den_arr = q * (p - n + Nd+ - Na-)
                // And understand *why* it's small.
                // Your current `initial_n` and `initial_p` are *designed* to make (p - n + Nd - Na) small.

                // A common strategy for initialization of a *depletion region* is to
                // assume that n and p are very low in the "middle" region and then
                // let the dopants define the charge.

                // Let's try a different approach to defining initial_n and initial_p
                // that allows for charge in the depletion region.
                // This still relies on a *conceptual* depletion region.

                amrex::Real phi_bi = (kb * T / q) * std::log((donor_doping * acceptor_doping) / (ni * ni)); // Built-in potential approximation

                // A linear or tanh-smoothed potential profile across the junction
                // This is a rough guess for the potential.
                amrex::Real initial_potential;
                if (cell_z < z_junction) { // N-side
                    initial_potential = 0.5 * phi_bi * (1.0 - std::tanh((z_junction - cell_z) / (junction_width)));
                } else { // P-side
                    initial_potential = -0.5 * phi_bi * (1.0 - std::tanh((cell_z - z_junction) / (junction_width)));
                }
                // Adjust potential to be relative to some reference (e.g., intrinsic Fermi level)
                // For simplicity, let's assume the N-side is higher potential, P-side lower.
                // Potential on N side should be higher than P side by phi_bi.
                // So, potential goes from +phi_bi/2 to -phi_bi/2 across the junction.

                // Let's refine the initial potential guess based on typically assumed potential for an ideal junction
                // N-side (far from junction) potential ~ 0 (or some reference)
                // P-side (far from junction) potential ~ -phi_bi
                initial_potential = -0.5 * phi_bi * (1.0 + std::tanh((cell_z - z_junction) / junction_width));
                // This will go from near 0 (n-side) to near -phi_bi (p-side)
                // The sign convention for potential can vary; often e.g. n-side is +phi_bi/2 and p-side -phi_bi/2

                // Let's try another common convention: potential difference from intrinsic level
                // Assuming EF_n - E_i = kT ln(Nd/ni) and EF_p - E_i = -kT ln(Na/ni)
                // and for a p-n junction at equilibrium, EF is flat.
                // So, the potential should track the difference between Ei and EF.
                // E_i - E_F = -q*potential
                // Thus, potential = (E_F - E_i) / q

                // Let V_T = k_B * T / q;
                // Potential on N-side (far from junction) should be approx V_T * ln(donor_doping / ni)
                // Potential on P-side (far from junction) should be approx -V_T * ln(acceptor_doping / ni)
                amrex::Real V_T = kb * T / q;
                amrex::Real phi_n_bulk = V_T * std::log(donor_doping / ni);
                amrex::Real phi_p_bulk = -V_T * std::log(acceptor_doping / ni);

                // This combines the potential profile with the smooth doping.
                // The tanh term will make potential transition from phi_n_bulk to phi_p_bulk.
                initial_potential = 0.5 * (phi_n_bulk + phi_p_bulk) - 0.5 * (phi_n_bulk - phi_p_bulk) * std::tanh((cell_z - z_junction) / junction_width);

                // Store this initial potential (if you have a potential MultiFab)
                // potential_arr(i,j,k) = initial_potential; // You'd need a potential MultiFab

                // Now calculate n and p from this initial potential
                initial_n = ni * std::exp(initial_potential / V_T);
                initial_p = ni * std::exp(-initial_potential / V_T);

                // Assign carrier concentrations to MultiFabs
                hole_den_arr(i,j,k) = initial_p;
                e_den_arr(i,j,k) = initial_n;

                // Assign doping concentrations to MultiFabs
                acceptor_den_arr(i,j,k) = Na_val;
                donor_den_arr(i,j,k) = Nd_val;

            } else { // Non-semiconductor regions
                hole_den_arr(i,j,k) = 0.0;
                e_den_arr(i,j,k) = 0.0;
                acceptor_den_arr(i,j,k) = 0.0;
                donor_den_arr(i,j,k) = 0.0;
            }

            // Calculate the charge density for Poisson's RHS
            // q is the elementary charge (e.g., 1.602e-19 C)
          //  charge_den_arr(i,j,k) = q * (hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));
        });
	/*
	amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real Na_val, Nd_val; // Use temporary values for N_A and N_D for this cell
            amrex::Real initial_n, initial_p;

            // Get the physical coordinate of the cell center in z
            Real cell_z = prob_lo[2] + (k+0.5) * dx[2];

            // Define junction parameters (these should be passed in or be global constants)
            const amrex::Real z_junction = 0.5e-6; // Center of the p-n junction
            const amrex::Real junction_width = 0.05e-6; // Characteristic width of the transition (adjust as needed)

            // SC region (mask >= 2.0 indicates semiconductor)
            if (mask(i,j,k) >= 2.0) {

                // Apply smooth doping profile for p-n junction
                // Using tanh for a smooth transition from n-type to p-type
                // At z_junction, Na_val and Nd_val should be roughly equal (and half of max doping)
                // For z < z_junction (n-type side), Nd_val should be high, Na_val low
                // For z > z_junction (p-type side), Na_val should be high, Nd_val low

                // Nd_val will be high for z < z_junction
                Nd_val = 0.5 * donor_doping * (1.0 - std::tanh((cell_z - z_junction) / junction_width));
                // Na_val will be high for z > z_junction
                Na_val = 0.5 * acceptor_doping * (1.0 + std::tanh((cell_z - z_junction) / junction_width));

                if (mask(i,j,k) == 2.0) { // If it's explicitly marked as intrinsic in the mask, keep it intrinsic
                    Na_val = 0.0;
                    Nd_val = 0.0;
                }


                // Calculate initial carrier concentrations based on local doping.
                // This assumes charge neutrality and full ionization at equilibrium.
                amrex::Real net_doping = Nd_val - Na_val;
                amrex::Real ni = intrinsic_carrier_concentration;

                if (net_doping > 0) { // n-type effective region
                    initial_n = 0.5 * (net_doping + std::sqrt(net_doping * net_doping + 4.0 * ni * ni));
                    initial_p = ni * ni / initial_n;
                } else if (net_doping < 0) { // p-type effective region
                    initial_p = 0.5 * (-net_doping + std::sqrt(net_doping * net_doping + 4.0 * ni * ni));
                    initial_n = ni * ni / initial_p;
                } else { // intrinsic (or very close to intrinsic)
                    initial_n = ni;
                    initial_p = ni;
                }

                // Assign initial carrier concentrations to MultiFabs
                hole_den_arr(i,j,k) = initial_p;
                e_den_arr(i,j,k) = initial_n;

                // Assign doping concentrations to MultiFabs
                acceptor_den_arr(i,j,k) = Na_val;
                donor_den_arr(i,j,k) = Nd_val;

            } else { // Non-semiconductor regions (e.g., oxide, metal contacts if you have them)
                // Set carrier and doping densities to zero outside SC region
                hole_den_arr(i,j,k) = 0.0;
                e_den_arr(i,j,k) = 0.0;
                acceptor_den_arr(i,j,k) = 0.0;
                donor_den_arr(i,j,k) = 0.0;
           //     charge_den_arr(i,j,k) = 0.0;
            }

            // Calculate the charge density for Poisson's RHS
            // q is the elementary charge (e.g., 1.602e-19 C)
            //charge_den_arr(i,j,k) = q * (hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));

            // if(i == 32 && j == 32 && k == 32) amrex::Print() << "hole_den_arr = " << hole_den_arr(i,j,k) << "\n" << "e_den_arr = " << e_den_arr(i,j,k) << "\n" << "acceptor_den_arr = " << acceptor_den_arr(i,j,k) << "\n" << "donor_den_arr = " << donor_den_arr(i,j,k) << "\n" << "charge_den_arr = " << charge_den_arr(i,j,k) << "\n";
        });
*/
        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            amrex::Real Na_val, Nd_val; // Use temporary values for N_A and N_D for this cell
            amrex::Real initial_n, initial_p;

            // SC region (mask >= 2.0 indicates semiconductor)
            if (mask(i,j,k) >= 2.0) {

                if (mask(i,j,k) == 2.0) { // intrinsic
                    Na_val = 0.0;
                    Nd_val = 0.0;
                    initial_n = intrinsic_carrier_concentration;
                    initial_p = intrinsic_carrier_concentration;
                } else if (mask(i,j,k) == 3.0) { // p-type
                    Na_val = acceptor_doping;
                    Nd_val = 0.0;
                    // In p-type, p is majority, n is minority
                    initial_p = acceptor_doping; // Assume full ionization and charge neutrality
                    initial_n = intrinsic_carrier_concentration * intrinsic_carrier_concentration / initial_p;
                } else if (mask(i,j,k) == 4.0) { // n-type
                    Na_val = 0.0;
                    Nd_val = donor_doping;
                    // In n-type, n is majority, p is minority
                    initial_n = donor_doping; // Assume full ionization and charge neutrality
                    initial_p = intrinsic_carrier_concentration * intrinsic_carrier_concentration / initial_n;
                }

                // Assign initial carrier concentrations to MultiFabs
                hole_den_arr(i,j,k) = initial_p;
                e_den_arr(i,j,k) = initial_n;

                // Assign doping concentrations to MultiFabs
                acceptor_den_arr(i,j,k) = Na_val;
                donor_den_arr(i,j,k) = Nd_val;

            } else { // Non-semiconductor regions (e.g., oxide, metal contacts if you have them)
                // Set carrier and doping densities to zero outside SC region
                hole_den_arr(i,j,k) = 0.0;
                e_den_arr(i,j,k) = 0.0;
                acceptor_den_arr(i,j,k) = 0.0;
                donor_den_arr(i,j,k) = 0.0;
            }

            // Calculate the charge density for Poisson's RHS
            // q is the elementary charge (e.g., 1.602e-19 C)
 //           charge_den_arr(i,j,k) = q * (hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));

            // if(i == 32 && j == 32 && k == 32) amrex::Print() << "hole_den_arr = " << hole_den_arr(i,j,k) << "\n" << "e_den_arr = " << e_den_arr(i,j,k) << "\n" << "acceptor_den_arr = " << acceptor_den_arr(i,j,k) << "\n" << "donor_den_arr = " << donor_den_arr(i,j,k) << "\n" << "charge_den_arr = " << charge_den_arr(i,j,k) << "\n";
        });
//	*/
    }
    // Fill boundaries for all MultiFabs
    e_den.FillBoundary(geom.periodicity());
    p_den.FillBoundary(geom.periodicity());
    acceptor_den.FillBoundary(geom.periodicity());
    donor_den.FillBoundary(geom.periodicity());

    for (MFIter mfi(rho); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox(); // Or mfi.tilebox()
        //const Box& bx = mfi.growntilebox(1);

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
            charge_den_arr(i,j,k) = q * (hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));
            //charge_den_arr(i,j,k) = q * (hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k));
        });
    }
    rho.FillBoundary(geom.periodicity());
 }

// create a mask filled with integers to idetify different material types
//void InitializeMaterialMask(MultiFab& MaterialMask, 
//		            const Geometry& geom, 
//			    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
//                            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi)
//{
//    // loop over boxes
//    for (MFIter mfi(MaterialMask); mfi.isValid(); ++mfi)
//    {
//        const Box& bx = mfi.growntilebox(MaterialMask.nGrow());
//        // extract dx from the geometry object
//        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
//
//        const Array4<Real>& mask = MaterialMask.array(mfi);
//
//
//        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
//        {
//             Real x = prob_lo[0] + (i+0.5) * dx[0];
//             Real y = prob_lo[1] + (j+0.5) * dx[1];
//             Real z = prob_lo[2] + (k+0.5) * dx[2];
//
//             //FE:0, DE:1, Source/Drain:2, p_type:3, n_type:4
//             if (x <= FE_hi[0] && x >= FE_lo[0] && y <= FE_hi[1] && y >= FE_lo[1] && z <= FE_hi[2] && z >= FE_lo[2]) {
//                 mask(i,j,k) = 0.;
//             } else if (x <= DE_hi[0] && x >= DE_lo[0] && y <= DE_hi[1] && y >= DE_lo[1] && z <= DE_hi[2] && z >= DE_lo[2]) {
//                 mask(i,j,k) = 1.;
//		 //intrinsic
//             } else if (x <= SC_hi[0] && x >= SC_lo[0] && y <= SC_hi[1] && y >= SC_lo[1] && z <= SC_hi[2] && z >= SC_lo[2]) {
//                 mask(i,j,k) = 2.;
//                //p_type
//	     } else if (x <= p_type_hi[0] && x >= p_type_lo[0] && y <= p_type_hi[1] && y >= p_type_lo[1] && z <= p_type_hi[2] && z >= p_type_lo[2]){
//                    mask(i,j,k) = 3.;
//                //n_type 
//	     } else if (x <= n_type_hi[0] && x >= n_type_lo[0] && y <= n_type_hi[1] && y >= n_type_lo[1] && z <= n_type_hi[2] && z >= n_type_lo[2]){
//                    mask(i,j,k) = 4.;
//             } else {
//	         mask(i,j,k) = 1.; //spacer is DE
//	     }
//        });
//    }
//    MaterialMask.FillBoundary(geom.periodicity());
//}
//
void InitializeMaterialMask(MultiFab& MaterialMask,
                            const Geometry& geom,
                            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi)
{

    for (MFIter mfi(MaterialMask); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(MaterialMask.nGrow());
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
        const Array4<Real>& mask = MaterialMask.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];

            if (x >= n_type_lo[0] && x <= n_type_hi[0] &&
                y >= n_type_lo[1] && y <= n_type_hi[1] &&
                z >= n_type_lo[2] && z <= n_type_hi[2])
            {
                mask(i,j,k) = 4.; // n_type
            }
            else if (x >= p_type_lo[0] && x <= p_type_hi[0] &&
                     y >= p_type_lo[1] && y <= p_type_hi[1] &&
                     z >= p_type_lo[2] && z <= p_type_hi[2])
            {
                mask(i,j,k) = 3.; // p_type
            }
            else if (x >= FE_lo[0] && x <= FE_hi[0] &&
                     y >= FE_lo[1] && y <= FE_hi[1] &&
                     z >= FE_lo[2] && z <= FE_hi[2])
            {
                mask(i,j,k) = 0.; // FE
            }
            else if (x >= DE_lo[0] && x <= DE_hi[0] &&
                     y >= DE_lo[1] && y <= DE_hi[1] &&
                     z >= DE_lo[2] && z <= DE_hi[2])
            {
                mask(i,j,k) = 1.; // DE
            }
            else if (x >= SC_lo[0] && x <= SC_hi[0] &&
                     y >= SC_lo[1] && y <= SC_hi[1] &&
                     z >= SC_lo[2] && z <= SC_hi[2])
            {
                mask(i,j,k) = 2.; // SC (Source/Drain)
            }
            else
            {
                mask(i,j,k) = 1.; // Default: Assign to DE (Dielectric) for any region not explicitly defined above
            }

        });
    }
    MaterialMask.FillBoundary(geom.periodicity());
}

// initialization of mask (device geometry) with parser
void InitializeMaterialMask(c_FerroX& rFerroX, const Geometry& geom, MultiFab& MaterialMask)
{ 
    auto& rGprop = rFerroX.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv = MaterialMask.ixType().toIntVect();

    for (MFIter mfi(MaterialMask, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& mask_arr = MaterialMask.array(mfi);
        const auto& bx = mfi.tilebox();

	std::string m_mask_s;
	std::unique_ptr<amrex::Parser> m_mask_parser;
        std::string m_str_device_geom_function;

	ParmParse pp_mask("device_geom");


	if (pp_mask.query("device_geom_function(x,y,z)", m_str_device_geom_function) ) {
            m_mask_s = "parse_device_geom_function";
        }

        if (m_mask_s == "parse_device_geom_function") {
            Store_parserString(pp_mask, "device_geom_function(x,y,z)", m_str_device_geom_function);
            m_mask_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_device_geom_function,{"x","y","z"}));
        }

        const auto& macro_parser = m_mask_parser->compile<3>();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv,macro_parser,mask_arr);
        });

    }
	MaterialMask.FillBoundary(geom.periodicity());
}

// initialization of t-phase mask with parser
void Initialize_tphase_Mask(c_FerroX& rFerroX, const Geometry& geom, MultiFab& tphaseMask)
{ 
    auto& rGprop = rFerroX.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv = tphaseMask.ixType().toIntVect();

    for (MFIter mfi(tphaseMask, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& mask_arr = tphaseMask.array(mfi);
        const auto& bx = mfi.tilebox();

	std::string tphase_mask_s;
	std::unique_ptr<amrex::Parser> tphase_mask_parser;
        std::string m_str_tphase_geom_function;

	ParmParse pp_mask("tphase_geom");


	if (pp_mask.query("tphase_geom_function(x,y,z)", m_str_tphase_geom_function) ) {
            tphase_mask_s = "parse_tphase_geom_function";
        }

        if (tphase_mask_s == "parse_tphase_geom_function") {
            Store_parserString(pp_mask, "tphase_geom_function(x,y,z)", m_str_tphase_geom_function);
            tphase_mask_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_tphase_geom_function,{"x","y","z"}));
        }

        const auto& macro_parser = tphase_mask_parser->compile<3>();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv,macro_parser,mask_arr);
        });

    }
	tphaseMask.FillBoundary(geom.periodicity());
}


// initialization of Euler angles
void Initialize_Euler_angles(c_FerroX& rFerroX, const Geometry& geom, MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta)
{ 
    auto& rGprop = rFerroX.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv_alpha = angle_alpha.ixType().toIntVect();
    const auto iv_beta = angle_beta.ixType().toIntVect();
    const auto iv_theta = angle_theta.ixType().toIntVect();

    for (MFIter mfi(angle_alpha, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& alpha_arr = angle_alpha.array(mfi);
        const auto& beta_arr = angle_beta.array(mfi);
        const auto& theta_arr = angle_theta.array(mfi);
        const auto& bx = mfi.tilebox();

	std::string alpha_s;
	std::unique_ptr<amrex::Parser> alpha_parser;
        std::string m_str_alpha_function;

	std::string beta_s;
	std::unique_ptr<amrex::Parser> beta_parser;
        std::string m_str_beta_function;

	std::string theta_s;
	std::unique_ptr<amrex::Parser> theta_parser;
        std::string m_str_theta_function;

	ParmParse pp_alpha("angle_alpha");


	if (pp_alpha.query("alpha_function(x,y,z)", m_str_alpha_function) ) {
            alpha_s = "parse_alpha_function";
        }

        if (alpha_s == "parse_alpha_function") {
            Store_parserString(pp_alpha, "alpha_function(x,y,z)", m_str_alpha_function);
            alpha_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_alpha_function,{"x","y","z"}));
        }

	ParmParse pp_beta("angle_beta");


	if (pp_beta.query("beta_function(x,y,z)", m_str_beta_function) ) {
            beta_s = "parse_beta_function";
        }

        if (beta_s == "parse_beta_function") {
            Store_parserString(pp_beta, "beta_function(x,y,z)", m_str_beta_function);
            beta_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_beta_function,{"x","y","z"}));
        }

	ParmParse pp_theta("angle_theta");


	if (pp_theta.query("theta_function(x,y,z)", m_str_theta_function) ) {
            theta_s = "parse_theta_function";
        }

        if (theta_s == "parse_theta_function") {
            Store_parserString(pp_theta, "theta_function(x,y,z)", m_str_theta_function);
            theta_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_theta_function,{"x","y","z"}));
        }

        const auto& macro_parser_alpha = alpha_parser->compile<3>();
        const auto& macro_parser_beta = beta_parser->compile<3>();
        const auto& macro_parser_theta = theta_parser->compile<3>();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_alpha,macro_parser_alpha,alpha_arr);
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_beta, macro_parser_beta, beta_arr );
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_theta,macro_parser_theta,theta_arr);
        });

    }
	angle_alpha.FillBoundary(geom.periodicity());
	angle_beta.FillBoundary(geom.periodicity());
	angle_theta.FillBoundary(geom.periodicity());
}

