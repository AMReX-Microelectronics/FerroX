#include "TotalEnergyDensity.H"
#include "DerivativeAlgorithm.H"

void CalculateTDGL_RHS(Array<MultiFab, AMREX_SPACEDIM> &GL_rhs,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                Array<MultiFab, AMREX_SPACEDIM> &E,
                MultiFab&                       Gamma,
                MultiFab&                 MaterialMask,
                MultiFab&                 tphaseMask,
                MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta,
                const Geometry& geom,
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi)
{
        // loop over boxes
        for ( MFIter mfi(P_old[0]); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real> &GL_RHS_x = GL_rhs[0].array(mfi);
            const Array4<Real> &GL_RHS_y = GL_rhs[1].array(mfi);
            const Array4<Real> &GL_RHS_z = GL_rhs[2].array(mfi);
            const Array4<Real> &pOld_x = P_old[0].array(mfi);
            const Array4<Real> &pOld_y = P_old[1].array(mfi);
            const Array4<Real> &pOld_z = P_old[2].array(mfi);
            const Array4<Real> &Ex = E[0].array(mfi);
            const Array4<Real> &Ey = E[1].array(mfi);
            const Array4<Real> &Ez = E[2].array(mfi);
            const Array4<Real>& Gam = Gamma.array(mfi);
            const Array4<Real>& mask = MaterialMask.array(mfi);
            const Array4<Real>& tphase = tphaseMask.array(mfi);

            const Array4<Real> &alpha_arr = angle_alpha.array(mfi);
            const Array4<Real> &beta_arr = angle_beta.array(mfi);
            const Array4<Real> &theta_arr = angle_theta.array(mfi);


            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real grad_term, phi_term, d2P_z;
                Real x    = prob_lo[0] + (i+0.5) * dx[0];
                Real x_hi = prob_lo[0] + (i+1.5) * dx[0];
                Real x_lo = prob_lo[0] + (i-0.5) * dx[0];

                Real y    = prob_lo[1] + (j+0.5) * dx[1];
                Real y_hi = prob_lo[1] + (j+1.5) * dx[1];
                Real y_lo = prob_lo[1] + (j-0.5) * dx[1];

                Real z    = prob_lo[2] + (k+0.5) * dx[2];
                Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

               //Convert Euler angles from degrees to radians 
               amrex::Real alpha_rad = 0.0174533*alpha_arr(i,j,k);
               amrex::Real beta_rad = 0.0174533*beta_arr(i,j,k);
               amrex::Real theta_rad = 0.0174533*theta_arr(i,j,k);
  
               amrex::Real R_11, R_12, R_13, R_21, R_22, R_23, R_31, R_32, R_33;
  
               if(use_Euler_angles){
                  R_11 = cos(alpha_rad)*cos(theta_rad) - cos(beta_rad)*sin(alpha_rad)*sin(theta_rad);  
                  R_12 = sin(alpha_rad)*cos(theta_rad) + cos(beta_rad)*cos(alpha_rad)*sin(theta_rad);  
                  R_13 = sin(beta_rad)*sin(theta_rad);  
                  R_21 = -cos(beta_rad)*cos(theta_rad)*sin(alpha_rad) - cos(alpha_rad)*sin(theta_rad);  
                  R_22 = cos(beta_rad)*cos(alpha_rad)*cos(theta_rad) - sin(alpha_rad)*sin(theta_rad);  
                  R_23 = sin(beta_rad)*cos(theta_rad);  
                  R_31 = sin(alpha_rad)*sin(beta_rad);  
                  R_32 = -cos(alpha_rad)*sin(beta_rad);  
                  R_33 = cos(beta_rad);  
               } else {
                  R_11 = cos(beta_rad)*cos(theta_rad);  
                  R_12 = sin(alpha_rad)*sin(beta_rad)*cos(theta_rad) - cos(alpha_rad)*sin(theta_rad);  
                  R_13 = cos(alpha_rad)*sin(beta_rad)*cos(theta_rad) + sin(alpha_rad)*sin(theta_rad);  
                  R_21 = cos(beta_rad)*sin(theta_rad);  
                  R_22 = sin(beta_rad)*sin(alpha_rad)*sin(theta_rad) + cos(alpha_rad)*cos(theta_rad);
                  R_23 = cos(alpha_rad)*sin(beta_rad)*sin(theta_rad) - sin(alpha_rad)*cos(theta_rad);
                  R_31 = -sin(beta_rad);
                  R_32 = sin(alpha_rad)*cos(beta_rad);
                  R_33 = cos(alpha_rad)*cos(beta_rad);
               }

                Real dFdPx_Landau = alpha*pOld_x(i,j,k) + beta*std::pow(pOld_x(i,j,k),3.) + FerroX::gamma*std::pow(pOld_x(i,j,k),5.)
                                    + 2. * alpha_12 * pOld_x(i,j,k) * std::pow(pOld_y(i,j,k),2.)
                                    + 2. * alpha_12 * pOld_x(i,j,k) * std::pow(pOld_z(i,j,k),2.)
                                    + 4. * alpha_112 * std::pow(pOld_x(i,j,k),3.) * (std::pow(pOld_y(i,j,k),2.) + std::pow(pOld_z(i,j,k),2.))
                                    + 2. * alpha_112 * pOld_x(i,j,k) * std::pow(pOld_y(i,j,k),4.)
                                    + 2. * alpha_112 * pOld_x(i,j,k) * std::pow(pOld_z(i,j,k),4.)
                                    + 2. * alpha_123 * pOld_x(i,j,k) * std::pow(pOld_y(i,j,k),2.) * std::pow(pOld_z(i,j,k),2.);

                Real dFdPy_Landau = alpha*pOld_y(i,j,k) + beta*std::pow(pOld_y(i,j,k),3.) + FerroX::gamma*std::pow(pOld_y(i,j,k),5.)
                                    + 2. * alpha_12 * pOld_y(i,j,k) * std::pow(pOld_x(i,j,k),2.)
                                    + 2. * alpha_12 * pOld_y(i,j,k) * std::pow(pOld_z(i,j,k),2.)
                                    + 4. * alpha_112 * std::pow(pOld_y(i,j,k),3.) * (std::pow(pOld_x(i,j,k),2.) + std::pow(pOld_z(i,j,k),2.))
                                    + 2. * alpha_112 * pOld_y(i,j,k) * std::pow(pOld_x(i,j,k),4.)
                                    + 2. * alpha_112 * pOld_y(i,j,k) * std::pow(pOld_z(i,j,k),4.)
                                    + 2. * alpha_123 * pOld_y(i,j,k) * std::pow(pOld_x(i,j,k),2.) * std::pow(pOld_z(i,j,k),2.);
                
                Real dFdPz_Landau = alpha*pOld_z(i,j,k) + beta*std::pow(pOld_z(i,j,k),3.) + FerroX::gamma*std::pow(pOld_z(i,j,k),5.)
                                    + 2. * alpha_12 * pOld_z(i,j,k) * std::pow(pOld_x(i,j,k),2.)
                                    + 2. * alpha_12 * pOld_z(i,j,k) * std::pow(pOld_y(i,j,k),2.)
                                    + 4. * alpha_112 * std::pow(pOld_z(i,j,k),3.) * (std::pow(pOld_x(i,j,k),2.) + std::pow(pOld_y(i,j,k),2.))
                                    + 2. * alpha_112 * pOld_z(i,j,k) * std::pow(pOld_x(i,j,k),4.)
                                    + 2. * alpha_112 * pOld_z(i,j,k) * std::pow(pOld_y(i,j,k),4.)
                                    + 2. * alpha_123 * pOld_z(i,j,k) * std::pow(pOld_x(i,j,k),2.) * std::pow(pOld_y(i,j,k),2.);

                Real dFdPx_grad = - g11 * DoubleDPDx(pOld_x, mask, i, j, k, dx)
                                  - (g44 + g44_p) * DoubleDPDy(pOld_x, mask, i, j, k, dx)
                                  - (g44 + g44_p) * DoubleDPDz(pOld_x, mask, i, j, k, dx)
                                  - (g12 + g44 - g44_p) * DoubleDPDxDy(pOld_y, mask, i, j, k, dx)  // d2P/dxdy
                                  - (g12 + g44 - g44_p) * DoubleDPDxDz(pOld_z, mask, i, j, k, dx); // d2P/dxdz
                
                Real dFdPy_grad = - g11 * DoubleDPDy(pOld_y, mask, i, j, k, dx)
                                  - (g44 - g44_p) * DoubleDPDx(pOld_y, mask, i, j, k, dx)
                                  - (g44 - g44_p) * DoubleDPDz(pOld_y, mask, i, j, k, dx)
                                  - (g12 + g44 + g44_p) * DoubleDPDxDy(pOld_x, mask, i, j, k, dx) // d2P/dxdy
                                  - (g12 + g44 - g44_p) * DoubleDPDyDz(pOld_z, mask, i, j, k, dx);// d2P/dydz

		//Switch g11 and g44 temporarily for multiphase simulations. This will be generalized later
                Real dFdPz_grad = - g44 * ( R_31*R_31*DoubleDPDx(pOld_z, mask, i, j, k, dx)
                                           +R_32*R_32*DoubleDPDy(pOld_z, mask, i, j, k, dx)
                                           +R_33*R_33*DoubleDPDz(pOld_z, mask, i, j, k, dx)
                                           +2.*R_31*R_32*DoubleDPDxDy(pOld_z, mask, i, j, k, dx)
                                           +2.*R_32*R_33*DoubleDPDyDz(pOld_z, mask, i, j, k, dx)
                                           +2.*R_33*R_31*DoubleDPDxDz(pOld_z, mask, i, j, k, dx))
                                           
                                  - (g11 - g44_p) * ( R_11*R_11*DoubleDPDx(pOld_z, mask, i, j, k, dx) 
                                                     +R_12*R_12*DoubleDPDy(pOld_z, mask, i, j, k, dx) 
                                                     +R_13*R_13*DoubleDPDz(pOld_z, mask, i, j, k, dx) 
                                                     +2.*R_11*R_12*DoubleDPDxDy(pOld_z, mask, i, j, k, dx) 
                                                     +2.*R_12*R_13*DoubleDPDyDz(pOld_z, mask, i, j, k, dx) 
                                                     +2.*R_13*R_11*DoubleDPDxDz(pOld_z, mask, i, j, k, dx))

                                  - (g44 - g44_p) * ( R_21*R_21*DoubleDPDx(pOld_z, mask, i, j, k, dx)
                                                     +R_22*R_22*DoubleDPDy(pOld_z, mask, i, j, k, dx)
                                                     +R_23*R_23*DoubleDPDz(pOld_z, mask, i, j, k, dx)
                                                     +2.*R_21*R_22*DoubleDPDxDy(pOld_z, mask, i, j, k, dx)
                                                     +2.*R_22*R_23*DoubleDPDyDz(pOld_z, mask, i, j, k, dx)
                                                     +2.*R_23*R_21*DoubleDPDxDz(pOld_z, mask, i, j, k, dx))

                                  - (g44 + g44_p + g12) * DoubleDPDyDz(pOld_y, mask, i, j, k, dx) // d2P/dydz
                                  - (g44 + g44_p + g12) * DoubleDPDxDz(pOld_x, mask, i, j, k, dx); // d2P/dxdz

                GL_RHS_x(i,j,k)  = -1.0 * Gam(i,j,k) *
                    (  dFdPx_Landau
                     + dFdPx_grad
		     - Ex(i,j,k)
                     //+ DFDx(phi, i, j, k, dx)
                    );

                GL_RHS_y(i,j,k)  = -1.0 * Gam(i,j,k) *
                    (  dFdPy_Landau
                     + dFdPy_grad
		     - Ey(i,j,k)
                     //+ DFDy(phi, i, j, k, dx)
                    );

		GL_RHS_x(i,j,k)  = 0.0;
		GL_RHS_y(i,j,k)  = 0.0;
                GL_RHS_z(i,j,k)  = -1.0 * Gam(i,j,k) *
                    (  dFdPz_Landau
                     + dFdPz_grad
		     - Ez(i,j,k)
                     //+ DphiDz(phi, z_hi, z_lo, i, j, k, dx, prob_lo, prob_hi)
                    );

		//set t_phase GL_RHS_z to zero so that it stays zero. It is initialized to zero in t-phase as well
                //if(x <= t_phase_hi[0] && x >= t_phase_lo[0] && y <= t_phase_hi[1] && y >= t_phase_lo[1] && z <= t_phase_hi[2] && z >= t_phase_lo[2]){
                if(tphase(i,j,k) == 1.0){
                  GL_RHS_z(i,j,k) = 0.0;
                }
            });
        }
}


