#include "TDGL.H"

// INITIALIZE rho in SC region
void InitializePandRho(MultiFab&   P_old,
                   MultiFab&   Gamma,
                   MultiFab&   rho,
                   MultiFab&   e_den,
                   MultiFab&   p_den,
                   Real        SC_lo,
                   Real        SC_hi,
                   Real        DE_lo,
                   Real        DE_hi,
		   Real        BigGamma,
                   Real        q,
                   Real        Ec,
                   Real        Ev,
                   Real        kb,
                   Real        T,
                   Real        Nc,
                   Real        Nv,
                   amrex::GpuArray<amrex::Real, 3> prob_lo,
                   amrex::GpuArray<amrex::Real, 3> prob_hi,
                   const       Geometry& geom)
{


    // loop over boxes
    for (MFIter mfi(rho); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();
        
	const Array4<Real>& pOld = P_old.array(mfi);
        const Array4<Real>& Gam = Gamma.array(mfi);

	Real pi = 3.141592653589793238;
        // set P
        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];
            if (z <= DE_hi) {
               pOld(i,j,k) = 0.0;
               Gam(i,j,k) = 0.0;
            } else {
               double tmp = (i%3 + j%2 + k%4)/6.;
               //pOld(i,j,k) = (-1.0 + 2.0*tmp)*0.002;
               //pOld(i,j,k) = (-1.0 + 2.0*Random())*0.002;
	       pOld(i,j,k) = 0.002*exp(-(x*x/(2.0*5.e-9*5.e-9) + y*y/(2.0*5.e-9*5.e-9) + (z-1.5*DE_hi)*(z - 1.5*DE_hi)/(2.0*2.0e-9*2.0e-9)));
	       //pOld(i,j,k) = 0.002*cos(2*pi*x/(prob_hi[0] - prob_lo[0]))*cos(2*pi*y/(prob_hi[1] - prob_lo[1]))*sin(2*pi*(z-DE_hi)/(prob_hi[2] - DE_hi));
               Gam(i,j,k) = BigGamma;
            }
        });

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev 

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
             Real z = prob_lo[2] + (k+0.5) * dx[2];

             if(z <= SC_hi){ //SC region

                //Real qPhi = 0.5*(Ec + Ev); //eV
                Real qPhi = 0.5*(Ec + Ev) - 0.56; //eV
                hole_den_arr(i,j,k) = Nv*exp(-(qPhi - Ev)*1.602e-19/(kb*T));
                e_den_arr(i,j,k) = Nc*exp(-(Ec - qPhi)*1.602e-19/(kb*T));
                charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));
                //if(e_den_arr(i,j,k) > 0) std::cout << "e_den(" <<i << "," <<  j << ", " << k <<" ) = "<< e_den_arr(i,j,k) << ", coeff = " << coeff << std::endl; 
	     } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
    // fill periodic ghost cells
    P_old.FillBoundary(geom.periodicity());

 }


// Compute rho in SC region for given phi
void ComputeRho(MultiFab&      PoissonPhi,
                MultiFab&      rho,
                MultiFab&      e_den,
                MultiFab&      p_den,
                Real           Sc_lo,
                Real           SC_hi,
                Real           q, Real Ec,
                Real           Ev,
                Real           kb,
                Real           T,
                Real           Nc,
                Real           Nv,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
                const          Geometry& geom)
{
    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

       // extract dx from the geometry object
       GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev 

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& phi = PoissonPhi.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
             Real z = prob_lo[2] + (k+0.5) * dx[2];

             if(z <= SC_hi){ //SC region

                hole_den_arr(i,j,k) = Nv*exp(-(q*phi(i,j,k) - Ev*1.602e-19)/(kb*T)); // Testing phi = 0 initialization
                e_den_arr(i,j,k) = Nc*exp(-(Ec*1.602e-19 - q*phi(i,j,k))/(kb*T)); // Testing phi = 0 initialization
                charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k)); // Testing phi = 0 initialization
                //charge_den_arr(i,j,k) = 0.0; // Testing rho = 0 initialization
             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
 }

void ComputePoissonRHS(MultiFab&               PoissonRHS,
                MultiFab&                      P_old,
                MultiFab&                      rho,
                Real                           FE_lo,
                Real                           FE_hi,
                Real                           DE_lo,
                Real                           DE_hi,
                Real                           SC_lo,
                Real                           SC_hi,
                int                            P_BC_flag_lo,
                int                            P_BC_flag_hi,
                Real                           lambda,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
                const Geometry&                 geom)
{
    for ( MFIter mfi(PoissonRHS); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();
            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real>& pOld = P_old.array(mfi);
            const Array4<Real>& RHS = PoissonRHS.array(mfi);
            const Array4<Real>& charge_den_arr = rho.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                 Real z    = prob_lo[2] + (k+0.5) * dx[2];
                 Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                 Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

                 if(z <= SC_hi){ //SC region

                   RHS(i,j,k) = charge_den_arr(i,j,k);

                 } else if(z < DE_hi){ //DE region

                   RHS(i,j,k) = 0.;

                 } else if (DE_hi > z_lo && DE_hi <= z) { //FE side of FE-DE interface

                   if(P_BC_flag_lo == 0){
                     Real P_int = 0.0;
                     RHS(i,j,k) = -(-4.*P_int + 3.*pOld(i,j,k) + pOld(i,j,k+1))/(3.*dx[2]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k+1)
                   } else if (P_BC_flag_lo == 1){
                     Real P_int = pOld(i,j,k)/(1 + dx[2]/2/lambda);
                     Real dPdz = P_int/lambda;
                     RHS(i,j,k) = -(dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/(2.*dx[2]);
                   } else if (P_BC_flag_lo == 2){
                     Real dPdz = 0.;
                     RHS(i,j,k) = -(dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/(2.*dx[2]);
                   }

                 } else if (z_hi > prob_hi[2]){ //Top metal
                    
	             if(P_BC_flag_hi == 0){
                     Real P_int = 0.0;
                     RHS(i,j,k) = -(4.*P_int - 3.*pOld(i,j,k) - pOld(i,j,k-1))/(3.*dx[2]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k-1)
                     } else if (P_BC_flag_hi == 1){
                     Real P_int = pOld(i,j,k)/(1 - dx[2]/2/lambda);
                     Real dPdz = P_int/lambda;
                     RHS(i,j,k) = -(dx[2]*dPdz + pOld(i,j,k) - pOld(i,j,k-1))/(2.*dx[2]);
                     } else if (P_BC_flag_hi == 2){
                     Real dPdz = 0.;
                     RHS(i,j,k) = -(dx[2]*dPdz + pOld(i,j,k) - pOld(i,j,k-1))/(2.*dx[2]);
                   }

                 }else{ //inside FE

                   RHS(i,j,k) = -(pOld(i,j,k+1) - pOld(i,j,k-1))/(2.*dx[2]);

                 }

            });
        }
   
}

void CalculateTDGL_RHS(MultiFab&                GL_rhs,
                MultiFab&                       P_old,
                MultiFab&                       PoissonPhi,
                MultiFab&                       Gamma,
                Real                            FE_lo,
                Real                            FE_hi,
                Real                            DE_lo,
                Real                            DE_hi,
                Real                            SC_lo,
                Real                            SC_hi,
                int                             P_BC_flag_lo,
                int                             P_BC_flag_hi,
                int                             Phi_Bc_lo,
                int                             Phi_Bc_hi,
                Real                            alpha,
                Real                            beta,
                Real                            gamma,
                Real                            g11,
                Real                            g44,
                Real                            lambda,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
                const Geometry& geom)
{
	// loop over boxes
        for ( MFIter mfi(P_old); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real>& GL_RHS = GL_rhs.array(mfi);
            const Array4<Real>& pOld = P_old.array(mfi);
            const Array4<Real>& phi = PoissonPhi.array(mfi);
            const Array4<Real>& Gam = Gamma.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real grad_term, phi_term, d2P_z;
                Real z    = prob_lo[2] + (k+0.5) * dx[2];
                Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

		if(z_lo < prob_lo[2]){ //Bottom metal

                  grad_term = 0.0;
                  phi_term = (-4.*Phi_Bc_lo + 3.*phi(i,j,k) + phi(i,j,k+1))/(3.*dx[2]);
                  //phi_term = (phi(i,j,k+1) - phi(i,j,k)) / (dx[2]);

                } else if(z < DE_hi){ //Below FE-DE interface

                  grad_term = 0.0;
                  phi_term = (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2]);

                } else if (DE_hi > z_lo && DE_hi <= z) { //FE side of FE-DE interface

                  if(P_BC_flag_lo == 0){
                    Real P_int = 0.0;
                    d2P_z = 4.*(2.*P_int - 3.*pOld(i,j,k) + pOld(i,j,k+1))/3./dx[2]/dx[2];//2nd Order
                  } else if (P_BC_flag_lo == 1){
                    Real P_int = pOld(i,j,k)/(1 + dx[2]/2/lambda);
                    Real dPdz = P_int/lambda;
                    d2P_z = (-dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/dx[2]/dx[2];//2nd Order
                  } else if (P_BC_flag_lo == 2){
                    Real dPdz = 0.;
                    d2P_z = (-dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/dx[2]/dx[2];//2nd Order
                  }

                  grad_term = g11 * d2P_z;
                  phi_term = (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2]);

                } else if (z_hi > prob_hi[2]){ //Top metal

			if(P_BC_flag_hi == 0){
                    Real P_int = 0.0;
                    d2P_z = 4.*(2.*P_int - 3.*pOld(i,j,k) + pOld(i,j,k-1))/3./dx[2]/dx[2];//2nd Order
                  } else if (P_BC_flag_hi == 1){
                    Real P_int = pOld(i,j,k)/(1 - dx[2]/2/lambda);
                    Real dPdz = P_int/lambda;
                    d2P_z = (dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k-1))/dx[2]/dx[2];//2nd Order
                  } else if (P_BC_flag_hi == 2){
                    Real dPdz = 0.;
                    d2P_z = (dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k-1))/dx[2]/dx[2];//2nd Order
                  }

                  grad_term = g11 * d2P_z;
                  phi_term = (4.*Phi_Bc_hi - 3.*phi(i,j,k) - phi(i,j,k-1))/(3.*dx[2]);

                }else{ //inside FE

                  grad_term = g11 * (pOld(i,j,k+1) - 2.*pOld(i,j,k) + pOld(i,j,k-1)) / (dx[2]*dx[2]);
                  phi_term = (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2]);

                }

		GL_RHS(i,j,k)  = -1.0 * Gam(i,j,k) *
                    (  alpha*pOld(i,j,k) + beta*std::pow(pOld(i,j,k),3.) + gamma*std::pow(pOld(i,j,k),5.)
                     - g44 * (pOld(i+1,j,k) - 2.*pOld(i,j,k) + pOld(i-1,j,k)) / (dx[0]*dx[0])
                     - g44 * (pOld(i,j+1,k) - 2.*pOld(i,j,k) + pOld(i,j-1,k)) / (dx[1]*dx[1])
                     - grad_term
                     + phi_term
                    ); 
            });
        }
}


void ComputeEfromPhi(MultiFab&                 PoissonPhi,
                MultiFab&                      Ex,
                MultiFab&                      Ey,
                MultiFab&                      Ez,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
                const Geometry&                 geom)
{
       // Calculate E from Phi

        for ( MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real>& Ex_arr = Ex.array(mfi);
            const Array4<Real>& Ey_arr = Ey.array(mfi);
            const Array4<Real>& Ez_arr = Ez.array(mfi);
            const Array4<Real>& phi = PoissonPhi.array(mfi);

            amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                     Ex_arr(i,j,k) = -(phi(i+1,j,k) - phi(i-1,j,k))/(2.*dx[0]);
                     Ey_arr(i,j,k) = -(phi(i,j+1,k) - phi(i,j-1,k))/(2.*dx[1]);

                     Real z    = prob_lo[2] + (k+0.5) * dx[2];
                     Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                     Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

                     if(z_lo < prob_lo[2]){ //Bottom Boundary
                       Ez_arr(i,j,k) = -(phi(i,j,k+1) - phi(i,j,k))/(dx[2]);
                     } else if (z_hi > prob_hi[2]){ //Top Boundary
                       Ez_arr(i,j,k) = -(phi(i,j,k) - phi(i,j,k-1))/(dx[2]);
                     }else{ //inside
                       Ez_arr(i,j,k) = -(phi(i,j,k+1) - phi(i,j,k-1))/(2.*dx[2]);
                     }
             });
        }

}


void InitializePermittivity(std::array< MultiFab, AMREX_SPACEDIM >& beta_face,
                Real                            FE_lo,
                Real                            FE_hi,
                Real                            DE_lo,
                Real                            DE_hi,
                Real                            SC_lo,
                Real                            SC_hi,
		Real 				epsilon_0,
		Real 				epsilonX_fe,
		Real 				epsilonZ_fe,
		Real 				epsilon_de,
		Real 				epsilon_si,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
                const Geometry&                 geom)
{

	    // set face-centered beta coefficient to
    // epsilon values in SC, FE, and DE layers
    // loop over boxes
    for (MFIter mfi(beta_face[0]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        const Array4<Real>& beta_f0 = beta_face[0].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          Real z = prob_lo[2] + (k+0.5) * dx[2];
          if(z <= SC_hi) {
             beta_f0(i,j,k) = epsilon_si * epsilon_0; //SC layer
          } else if(z <= DE_hi) {
             beta_f0(i,j,k) = epsilon_de * epsilon_0; //DE layer
          } else {
             beta_f0(i,j,k) = epsilonX_fe * epsilon_0; //FE layer
          }
        });
    }

    for (MFIter mfi(beta_face[1]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        const Array4<Real>& beta_f1 = beta_face[1].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          Real z = prob_lo[2] + (k+0.5) * dx[2];
          if(z <= SC_hi) {
             beta_f1(i,j,k) = epsilon_si * epsilon_0; //SC layer
          } else if(z <= DE_hi) {
             beta_f1(i,j,k) = epsilon_de * epsilon_0; //DE layer
          } else {
             beta_f1(i,j,k) = epsilonX_fe * epsilon_0; //FE layer
          }
        });
    }

    for (MFIter mfi(beta_face[2]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        const Array4<Real>& beta_f2 = beta_face[2].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          Real z = prob_lo[2] + (k+0.5) * dx[2];
          if(z < SC_hi) {
             beta_f2(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  }else if(z == SC_hi){
             beta_f2(i,j,k) = 0.5*(epsilon_de + epsilon_si) * epsilon_0; //SC-DE interface
          } else if(z < DE_hi) {
             beta_f2(i,j,k) = epsilon_de * epsilon_0; //DE layer
	  }else if(z == DE_hi){
             beta_f2(i,j,k) = 0.5*(epsilon_de + epsilonZ_fe) * epsilon_0; //DE-FE interface
          } else {
             beta_f2(i,j,k) = epsilonZ_fe * epsilon_0; //FE layer
          }
        });
    }

}

void SetPhiBC_z(MultiFab& PoissonPhi,
                amrex::GpuArray<int, 3> n_cell,
                Real                    Phi_Bc_lo,
                Real                    Phi_Bc_hi)
{
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(1);

        const Array4<Real>& Phi = PoissonPhi.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          if(k < 0) {
            Phi(i,j,k) = Phi_Bc_lo;
          } else if(k >= n_cell[2]){
            Phi(i,j,k) = Phi_Bc_hi;
          }
        });
    }
}
