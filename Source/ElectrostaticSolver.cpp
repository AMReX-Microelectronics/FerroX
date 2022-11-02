#include "ElectrostaticSolver.H"
#include "DerivativeAlgorithm.H"
#include "ChargeDensity.H"

void ComputePoissonRHS(MultiFab&               PoissonRHS,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                MultiFab&                      rho,
                const Geometry&                 geom)
{
    for ( MFIter mfi(PoissonRHS); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();
            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real> &pOld_x = P_old[0].array(mfi);
            const Array4<Real> &pOld_y = P_old[1].array(mfi);
            const Array4<Real> &pOld_z = P_old[2].array(mfi);
            const Array4<Real>& RHS = PoissonRHS.array(mfi);
            const Array4<Real>& charge_den_arr = rho.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                 Real z    = FerroX::prob_lo[2] + (k+0.5) * dx[2];
                 Real z_hi = FerroX::prob_lo[2] + (k+1.5) * dx[2];
                 Real z_lo = FerroX::prob_lo[2] + (k-0.5) * dx[2];

                 if(z <= FerroX::SC_hi){ //SC region

                   RHS(i,j,k) = charge_den_arr(i,j,k);

                 } else if(z < FerroX::DE_hi){ //DE region

                   RHS(i,j,k) = 0.;

                 } else {
                   RHS(i,j,k) = - DPDz(pOld_z, z, z_hi, z_lo, i, j, k, dx)
                                - DFDx(pOld_x, i, j, k, dx)
                                - DFDy(pOld_y, i, j, k, dx);

                 }

            });
        }
   
}

void dF_dPhi(MultiFab&            alpha_cc,
             MultiFab&            PoissonRHS, 
             MultiFab&            PoissonPhi, 
	     Array<MultiFab, AMREX_SPACEDIM>& P_old,
             MultiFab&            rho,
             MultiFab&            e_den,
             MultiFab&            p_den,
             const          Geometry& geom)

{
   
        MultiFab PoissonPhi_plus_delta(PoissonPhi.boxArray(), PoissonPhi.DistributionMap(), 1, 0); 
        MultiFab PoissonRHS_phi_plus_delta(PoissonRHS.boxArray(), PoissonRHS.DistributionMap(), 1, 0); 
 
        MultiFab::Copy(PoissonPhi_plus_delta, PoissonPhi, 0, 0, 1, 0); 
        PoissonPhi_plus_delta.plus(FerroX::delta, 0, 1, 0); 

        // Calculate rho from Phi in SC region
        ComputeRho(PoissonPhi_plus_delta, rho, e_den, p_den, geom);

        //Compute RHS of Poisson equation
        ComputePoissonRHS(PoissonRHS_phi_plus_delta, P_old, rho, geom);

        MultiFab::LinComb(alpha_cc, 1./FerroX::delta, PoissonRHS_phi_plus_delta, 0, -1./FerroX::delta, PoissonRHS, 0, 0, 1, 0);
}
void ComputePoissonRHS_Newton(MultiFab& PoissonRHS, 
                              MultiFab& PoissonPhi, 
                              MultiFab& alpha_cc)
{
     
        for ( MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real>& phi = PoissonPhi.array(mfi);
            const Array4<Real>& poissonRHS = PoissonRHS.array(mfi);
            const Array4<Real>& alpha = alpha_cc.array(mfi);

            amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                   poissonRHS(i,j,k) = poissonRHS(i,j,k) - alpha(i,j,k)*phi(i,j,k) ;
            });
        }
}

void ComputeEfromPhi(MultiFab&                 PoissonPhi,
                MultiFab&                      Ex,
                MultiFab&                      Ey,
                MultiFab&                      Ez,
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
                     Real z    = FerroX::prob_lo[2] + (k+0.5) * dx[2];
                     Real z_hi = FerroX::prob_lo[2] + (k+1.5) * dx[2];
                     Real z_lo = FerroX::prob_lo[2] + (k-0.5) * dx[2];

                     Ex_arr(i,j,k) = - DFDx(phi, i, j, k, dx);
                     Ey_arr(i,j,k) = - DFDy(phi, i, j, k, dx);
                     Ez_arr(i,j,k) = - DphiDz(phi, z, z_hi, z_lo, i, j, k, dx);

             });
        }

}


void InitializePermittivity(std::array< MultiFab, AMREX_SPACEDIM >& beta_face,
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
          Real z = FerroX::prob_lo[2] + (k+0.5) * dx[2];
          if(z < FerroX::SC_hi-small) {
             beta_f0(i,j,k) = FerroX::epsilon_si * FerroX::epsilon_0; //SC layer
	  } else if(z >= FerroX::SC_hi-small && z < FerroX::SC_hi+small){
             beta_f0(i,j,k) = 0.5*(FerroX::epsilon_de + FerroX::epsilon_si) * FerroX::epsilon_0; //SC-DE interface
          } else if(z < FerroX::DE_hi-small) {
             beta_f0(i,j,k) = FerroX::epsilon_de * FerroX::epsilon_0; //DE layer
	  } else if(z >= FerroX::DE_hi-small && z < FerroX::DE_hi+small){
             beta_f0(i,j,k) = 0.5*(FerroX::epsilon_de + FerroX::epsilonX_fe) * FerroX::epsilon_0; //DE-FE interface
             //beta_f0(i,j,k) = FerroX::epsilon_de * FerroX::epsilon_0; //DE-FE interface
          } else {
             beta_f0(i,j,k) = FerroX::epsilonX_fe * FerroX::epsilon_0; //FE layer
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
          Real z = FerroX::prob_lo[2] + (k+0.5) * dx[2];
          if(z < FerroX::SC_hi-small) {
             beta_f1(i,j,k) = FerroX::epsilon_si * FerroX::epsilon_0; //SC layer
	  } else if(z >= FerroX::SC_hi-small && z < FerroX::SC_hi+small){
             beta_f1(i,j,k) = 0.5*(FerroX::epsilon_de + FerroX::epsilon_si) * FerroX::epsilon_0; //SC-DE interface
          } else if(z < FerroX::DE_hi-small) {
             beta_f1(i,j,k) = FerroX::epsilon_de * FerroX::epsilon_0; //DE layer
	  } else if(z >= FerroX::DE_hi-small && z < FerroX::DE_hi+small){
             //beta_f1(i,j,k) = FerroX::epsilon_de * FerroX::epsilon_0; //DE-FE interface
             beta_f1(i,j,k) = 0.5*(FerroX::epsilon_de + FerroX::epsilonX_fe) * FerroX::epsilon_0; //DE-FE interface
          } else {
             beta_f1(i,j,k) = FerroX::epsilonX_fe * FerroX::epsilon_0; //FE layer
          }
        });
    }

    for (MFIter mfi(beta_face[2]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& beta_f2 = beta_face[2].array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          Real z = FerroX::prob_lo[2] + k * dx[2];
          if(z < FerroX::SC_hi-small) {
             beta_f2(i,j,k) = FerroX::epsilon_si * FerroX::epsilon_0; //SC layer
	  } else if(z >= FerroX::SC_hi-small && z < FerroX::SC_hi+small && FerroX::SC_hi > FerroX::prob_lo[2]+small){
             beta_f2(i,j,k) = 0.5*(FerroX::epsilon_de + FerroX::epsilon_si) * FerroX::epsilon_0; //SC-DE interface
          } else if(z < FerroX::DE_hi-small) {
             beta_f2(i,j,k) = FerroX::epsilon_de * FerroX::epsilon_0; //DE layer
	  } else if(z >= FerroX::DE_hi-small && z < FerroX::DE_hi+small){
             beta_f2(i,j,k) = 0.5*(FerroX::epsilon_de + FerroX::epsilonZ_fe) * FerroX::epsilon_0; //DE-FE interface
             //beta_f2(i,j,k) = FerroX::epsilon_de * FerroX::epsilon_0; //DE-FE interface
          } else {
             beta_f2(i,j,k) = FerroX::epsilonZ_fe * FerroX::epsilon_0; //FE layer
          }
        });
    }

}

void SetPhiBC_z(MultiFab& PoissonPhi)
{
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(1);

        const Array4<Real>& Phi = PoissonPhi.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
          if(k < 0) {
            Phi(i,j,k) = FerroX::Phi_Bc_lo;
          } else if(k >= FerroX::n_cell[2]){
            Phi(i,j,k) = FerroX::Phi_Bc_hi;
          }
        });
    }
}
