#include "ElectrostaticSolver.H"

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

                 } else{ //inside FE

                   RHS(i,j,k) = -(pOld(i,j,k+1) - pOld(i,j,k-1))/(2.*dx[2]);

                 }

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
                     } else{ //inside
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
    // extract dx from the geometry object
    GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

    Real small = dx[2]*1.e-6;
    
    // set face-centered beta coefficient to
    // epsilon values in SC, FE, and DE layers
    // loop over boxes
    for (MFIter mfi(beta_face[0]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& beta_f0 = beta_face[0].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          //Real z = prob_lo[2] + k * dx[2];
          Real z = prob_lo[2] + (k+0.5) * dx[2];
          if(z < SC_hi-small) {
             beta_f0(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  } else if(z >= SC_hi-small && z < SC_hi+small){
             beta_f0(i,j,k) = 0.5*(epsilon_de + epsilon_si) * epsilon_0; //SC-DE interface
          } else if(z < DE_hi-small) {
             beta_f0(i,j,k) = epsilon_de * epsilon_0; //DE layer
	  } else if(z >= DE_hi-small && z < DE_hi+small){
             beta_f0(i,j,k) = 0.5*(epsilon_de + epsilonX_fe) * epsilon_0; //DE-FE interface
             //beta_f0(i,j,k) = epsilon_de * epsilon_0; //DE-FE interface
          } else {
             beta_f0(i,j,k) = epsilonX_fe * epsilon_0; //FE layer
          }
        });
    }

    for (MFIter mfi(beta_face[1]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& beta_f1 = beta_face[1].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          //Real z = prob_lo[2] + k * dx[2];
          Real z = prob_lo[2] + (k+0.5) * dx[2];
          if(z < SC_hi-small) {
             beta_f1(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  } else if(z >= SC_hi-small && z < SC_hi+small){
             beta_f1(i,j,k) = 0.5*(epsilon_de + epsilon_si) * epsilon_0; //SC-DE interface
          } else if(z < DE_hi-small) {
             beta_f1(i,j,k) = epsilon_de * epsilon_0; //DE layer
	  } else if(z >= DE_hi-small && z < DE_hi+small){
             //beta_f1(i,j,k) = epsilon_de * epsilon_0; //DE-FE interface
             beta_f1(i,j,k) = 0.5*(epsilon_de + epsilonX_fe) * epsilon_0; //DE-FE interface
          } else {
             beta_f1(i,j,k) = epsilonX_fe * epsilon_0; //FE layer
          }
        });
    }

    for (MFIter mfi(beta_face[2]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& beta_f2 = beta_face[2].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          Real z = prob_lo[2] + k * dx[2];
          if(z < SC_hi-small) {
             beta_f2(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  } else if(z >= SC_hi-small && z < SC_hi+small && SC_hi > prob_lo[2]+small){
             beta_f2(i,j,k) = 0.5*(epsilon_de + epsilon_si) * epsilon_0; //SC-DE interface
          } else if(z < DE_hi-small) {
             beta_f2(i,j,k) = epsilon_de * epsilon_0; //DE layer
	  } else if(z >= DE_hi-small && z < DE_hi+small){
             beta_f2(i,j,k) = 0.5*(epsilon_de + epsilonZ_fe) * epsilon_0; //DE-FE interface
             //beta_f2(i,j,k) = epsilon_de * epsilon_0; //DE-FE interface
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
