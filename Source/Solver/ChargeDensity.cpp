#include "ChargeDensity.H"
#include "DerivativeAlgorithm.H"

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

// Drift-Diffusion : Compute rho in SC region for given phi, Jn, and Jp
void ComputeRho_DriftDiffusion(MultiFab&      PoissonPhi,
                               Array<MultiFab, AMREX_SPACEDIM> &Jn,
                               Array<MultiFab, AMREX_SPACEDIM> &Jp,
                               MultiFab&      rho,
                               MultiFab&      e_den,
                               MultiFab&      p_den,
                               MultiFab&      e_den_old,
                               MultiFab&      p_den_old,
                               const Geometry& geom,
		               const MultiFab& MaterialMask)
{
    MultiFab Ec_mf(rho.boxArray(), rho.DistributionMap(), 1, 0);
    MultiFab Ev_mf(rho.boxArray(), rho.DistributionMap(), 1, 0);
        
    //Calculate Ec and Ev
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& phi = PoissonPhi.array(mfi);
        const Array4<Real>& Ec_arr = Ec_mf.array(mfi);
        const Array4<Real>& Ev_arr = Ev_mf.array(mfi);
        const Array4<Real const>& mask = MaterialMask.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

             if (mask(i,j,k) >= 2.0) {
          	//electrons
                Ec_arr(i,j,k) = -q*phi(i,j,k) - electron_affinity;
                Ev_arr(i,j,k) = Ec_arr(i,j,k) - bandgap;
             } else {
                Ec_arr(i,j,k) = 0.0;
                Ev_arr(i,j,k) = 0.0;
             }
        });
    }

    // Calculate currents Jn and Jp
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& phi = PoissonPhi.array(mfi);
        const Array4<Real const>& mask = MaterialMask.array(mfi);

        const Array4<Real>& Ec_arr = Ec_mf.array(mfi);
        const Array4<Real>& Ev_arr = Ev_mf.array(mfi);
        const Array4<Real>& Jnx_arr = Jn[0].array(mfi);
        const Array4<Real>& Jny_arr = Jn[1].array(mfi);
        const Array4<Real>& Jnz_arr = Jn[2].array(mfi);

        const Array4<Real>& Jpx_arr = Jp[0].array(mfi);
        const Array4<Real>& Jpy_arr = Jp[1].array(mfi);
        const Array4<Real>& Jpz_arr = Jp[2].array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

             if (mask(i,j,k) >= 2.0) {

		 //electrons
                 Real gradEc_x = DFDx(Ec_arr, i, j, k, dx);
                 Real gradEc_y = DFDy(Ec_arr, i, j, k, dx);
                 Real gradEc_z = DFDz(Ec_arr, i, j, k, dx);
                 
                 Real gradn_x = DFDx(e_den_arr, i, j, k, dx);
                 Real gradn_y = DFDy(e_den_arr, i, j, k, dx);
                 Real gradn_z = DFDz(e_den_arr, i, j, k, dx);
                 
                 Jnx_arr(i,j,k) = electron_mobility*e_den_arr(i,j,k)*gradEc_x + kb*T*electron_mobility*gradn_x;
                 Jny_arr(i,j,k) = electron_mobility*e_den_arr(i,j,k)*gradEc_y + kb*T*electron_mobility*gradn_y;
                 Jnz_arr(i,j,k) = electron_mobility*e_den_arr(i,j,k)*gradEc_z + kb*T*electron_mobility*gradn_z;

		 //holes
                 Real gradEv_x = DFDx(Ev_arr, i, j, k, dx);
                 Real gradEv_y = DFDy(Ev_arr, i, j, k, dx);
                 Real gradEv_z = DFDz(Ev_arr, i, j, k, dx);
                 
                 Real gradp_x = DFDx(hole_den_arr, i, j, k, dx);
                 Real gradp_y = DFDy(hole_den_arr, i, j, k, dx);
                 Real gradp_z = DFDz(hole_den_arr, i, j, k, dx);
                 
                 Jpx_arr(i,j,k) = hole_mobility*hole_den_arr(i,j,k)*gradEv_x - kb*T*hole_mobility*gradp_x;
                 Jpy_arr(i,j,k) = hole_mobility*hole_den_arr(i,j,k)*gradEv_y - kb*T*hole_mobility*gradp_y;
                 Jpz_arr(i,j,k) = hole_mobility*hole_den_arr(i,j,k)*gradEv_z - kb*T*hole_mobility*gradp_z;

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

    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev
	MultiFab acceptor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
        MultiFab donor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
        MultiFab div_Jn(rho.boxArray(), rho.DistributionMap(), 1, 0);
        MultiFab div_Jp(rho.boxArray(), rho.DistributionMap(), 1, 0);

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& hole_den_old_arr = p_den.array(mfi);
        const Array4<Real>& e_den_old_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& phi = PoissonPhi.array(mfi);
	const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);
        const Array4<Real const>& mask = MaterialMask.array(mfi);

        const Array4<Real>& Jnx_arr = Jn[0].array(mfi);
        const Array4<Real>& Jny_arr = Jn[1].array(mfi);
        const Array4<Real>& Jnz_arr = Jn[2].array(mfi);
        const Array4<Real>& div_Jn_arr = div_Jn.array(mfi);

        const Array4<Real>& Jpx_arr = Jp[0].array(mfi);
        const Array4<Real>& Jpy_arr = Jp[1].array(mfi);
        const Array4<Real>& Jpz_arr = Jp[2].array(mfi);
        const Array4<Real>& div_Jp_arr = div_Jp.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

             if (mask(i,j,k) >= 2.0) {

                div_Jn_arr(i,j,k) = DFDx(Jnx_arr, i,j,k,dx) + DFDy(Jny_arr, i,j,k,dx) + DFDz(Jnz_arr, i,j,k,dx);
                div_Jp_arr(i,j,k) = DFDx(Jpx_arr, i,j,k,dx) + DFDy(Jpy_arr, i,j,k,dx) + DFDz(Jpz_arr, i,j,k,dx);

                e_den_arr(i,j,k) = e_den_old_arr(i,j,k) + dt*div_Jn_arr(i,j,k);
                hole_den_arr(i,j,k) = hole_den_old_arr(i,j,k) - dt*div_Jp_arr(i,j,k);

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

