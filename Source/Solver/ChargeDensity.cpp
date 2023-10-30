#include "ChargeDensity.H"

// Compute rho in SC region for given phi
void ComputeRho(MultiFab&      PoissonPhi,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                MultiFab&      rho,
                MultiFab&      e_den,
                MultiFab&      p_den,
		const MultiFab& MaterialMask,
                const Geometry& geom,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi)
{
    // Calculate average Pr. We take the average only over the FE region
    Real average_P_r = 0.;
    Real total_P_r = 0.;
    Real FE_index_counter = 0.;
    
    Compute_P_av(P_old, total_P_r, MaterialMask, FE_index_counter, average_P_r);

    //Calculate integrated electrode charge (Qe) based on eq 13 of https://pubs.aip.org/aip/jap/article/44/8/3379/6486/Depolarization-fields-in-thin-ferroelectric-films
    Real FE_thickness = FE_hi[2] - FE_lo[2];
    Real coth = (exp(2.*metal_thickness/metal_screening_length) + 1.0) / (exp(2.*metal_thickness/metal_screening_length) - 1.0);
    Real csch = (2.*exp(metal_thickness/metal_screening_length)) / (exp(2.*metal_thickness/metal_screening_length) - 1.0);
    Real numerator = 0.5 * FE_thickness * average_P_r / epsilonX_fe;
    Real denominator = metal_screening_length/epsilon_de*(coth - csch) + FE_thickness/(2.*epsilonX_fe);
    Real Qe = -numerator/denominator;

    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev
	MultiFab acceptor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
        MultiFab donor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);

        const Array4<Real> &pOld_p = P_old[0].array(mfi);
        const Array4<Real> &pOld_q = P_old[1].array(mfi);
        const Array4<Real> &pOld_r = P_old[2].array(mfi);

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& phi = PoissonPhi.array(mfi);
	const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);
        const Array4<Real const>& mask = MaterialMask.array(mfi);

 
        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

             Real z = prob_lo[2] + (k+0.5) * dx[2];
             Real z_metal = 0.;

             if (mask(i,j,k) >= 2.0) {

                if (mask(i,j,k) == 4.0) { //Metal
	    	
                   if(z <= FE_lo[2]){
                      z_metal = std::abs(FE_lo[2] - (k + 0.5) * dx[2]);
                   } else if (z >= FE_hi[2]){ 
                      z_metal = std::abs((k + 0.5) * dx[2] - FE_hi[2]);
                   }
                   charge_den_arr(i,j,k) = Qe/metal_screening_length*exp(-z_metal/metal_screening_length);
                
                } else {

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
                     Real n_0 = intrinsic_carrier_concentration;
                     Real p_0 = intrinsic_carrier_concentration;
                     hole_den_arr(i,j,k) = n_0*exp(-(q*phi(i,j,k))/(kb*T));
                     e_den_arr(i,j,k) =    p_0*exp(q*phi(i,j,k)/(kb*T));
                   }

		   //If in channel, set acceptor doping, else (Source/Drain) set donor doping
                   if (mask(i,j,k) == 3.0) {
                      acceptor_den_arr(i,j,k) = acceptor_doping;
                      donor_den_arr(i,j,k) = 0.0;
                   } else if (mask(i,j,k) == 2.0){ // Source / Drain
                      acceptor_den_arr(i,j,k) = 0.0;
                      donor_den_arr(i,j,k) = donor_doping;
                   }

		   charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));
                } 

             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
 }

void Compute_P_Sum(const std::array<MultiFab, AMREX_SPACEDIM>& P, Real& sum)
{

     // Initialize to zero
     sum = 0.;

     ReduceOps<ReduceOpSum> reduce_op;

     ReduceData<Real> reduce_data(reduce_op);
     using ReduceTuple = typename decltype(reduce_data)::Type;

     for (MFIter mfi(P[2],TilingIfNotGPU()); mfi.isValid(); ++mfi)
     {
         const Box& bx = mfi.tilebox();
         const Box& bx_grid = mfi.validbox();

         auto const& fab = P[2].array(mfi);

         reduce_op.eval(bx, reduce_data,
         [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
         {
             return {fab(i,j,k)};
         });
     }

     sum = amrex::get<0>(reduce_data.value());
     ParallelDescriptor::ReduceRealSum(sum);
}


void Compute_P_index_Sum(const MultiFab& MaterialMask, Real& count)
{

     // Initialize to zero
     count = 0.;

     ReduceOps<ReduceOpSum> reduce_op;

     ReduceData<Real> reduce_data(reduce_op);
     using ReduceTuple = typename decltype(reduce_data)::Type;

     for (MFIter mfi(MaterialMask, TilingIfNotGPU()); mfi.isValid(); ++mfi)
     {
         const Box& bx = mfi.tilebox();
         const Box& bx_grid = mfi.validbox();

         auto const& fab = MaterialMask.array(mfi);

         reduce_op.eval(bx, reduce_data,
         [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
         {
             if(fab(i,j,k) == 0.) {
               return {1.};
             } else {
               return {0.};
             }
            
         });
     }

     count = amrex::get<0>(reduce_data.value());
     ParallelDescriptor::ReduceRealSum(count);
}

void Compute_P_av(const std::array<MultiFab, AMREX_SPACEDIM>& P, Real& sum, const MultiFab& MaterialMask, Real& count, Real& P_av_z)
{
     Compute_P_Sum(P, sum);
     Compute_P_index_Sum(MaterialMask, count);
     P_av_z = sum/count;
}

