#include "TotalEnergyDensity.H"
#include "DerivativeAlgorithm.H"
#include "AMReX_CONSTANTS.H"


void CalculateTDGL_RHS(Array<MultiFab, AMREX_SPACEDIM> &GL_rhs,
                Array<MultiFab, AMREX_SPACEDIM> &GL_rhs_Landau,
                Array<MultiFab, AMREX_SPACEDIM> &GL_rhs_grad,
                Array<MultiFab, AMREX_SPACEDIM> &GL_rhs_elec,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                Array<MultiFab, AMREX_SPACEDIM> &E,
                MultiFab&                       Gamma,
                MultiFab&                 MaterialMask,
                MultiFab&                 tphaseMask,
                MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta,
                const Geometry& geom)
{
//        if (include_Landau == 1){
//           Calculate_Landau(GL_rhs_Landau, P_old, Gamma, tphaseMask);
//        }
//        if (include_Grad == 1){
//           Calculate_Grad(GL_rhs_grad, P_old, Gamma, MaterialMask, tphaseMask, angle_alpha, angle_beta, angle_theta, geom);
//        }
//        if (include_Elec == 1){
//           Calculate_Elec(GL_rhs_elec, E, Gamma, tphaseMask);
//        }

        // loop over boxes
        for ( MFIter mfi(P_old[0]); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real> &GL_RHS_p = GL_rhs[0].array(mfi);
            const Array4<Real> &GL_RHS_q = GL_rhs[1].array(mfi);
            const Array4<Real> &GL_RHS_r = GL_rhs[2].array(mfi);

            const Array4<Real> &GL_RHS_p_Landau = GL_rhs_Landau[0].array(mfi);
            const Array4<Real> &GL_RHS_q_Landau = GL_rhs_Landau[1].array(mfi);
            const Array4<Real> &GL_RHS_r_Landau = GL_rhs_Landau[2].array(mfi);

            const Array4<Real> &GL_RHS_p_grad = GL_rhs_grad[0].array(mfi);
            const Array4<Real> &GL_RHS_q_grad = GL_rhs_grad[1].array(mfi);
            const Array4<Real> &GL_RHS_r_grad = GL_rhs_grad[2].array(mfi);

            const Array4<Real> &GL_RHS_p_elec = GL_rhs_elec[0].array(mfi);
            const Array4<Real> &GL_RHS_q_elec = GL_rhs_elec[1].array(mfi);
            const Array4<Real> &GL_RHS_r_elec = GL_rhs_elec[2].array(mfi);


            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {

               if (include_Landau == 1){
                  GL_RHS_p(i,j,k) +=  GL_RHS_p_Landau(i,j,k); 
                  GL_RHS_q(i,j,k) +=  GL_RHS_q_Landau(i,j,k); 
                  GL_RHS_r(i,j,k) +=  GL_RHS_r_Landau(i,j,k); 
               }
               if (include_Grad == 1){
                  GL_RHS_p(i,j,k) +=  GL_RHS_p_grad(i,j,k); 
                  GL_RHS_q(i,j,k) +=  GL_RHS_q_grad(i,j,k); 
                  GL_RHS_r(i,j,k) +=  GL_RHS_r_grad(i,j,k); 
               }
               if (include_Elec == 1){
                  GL_RHS_p(i,j,k) +=  GL_RHS_p_elec(i,j,k); 
                  GL_RHS_q(i,j,k) +=  GL_RHS_q_elec(i,j,k); 
                  GL_RHS_r(i,j,k) +=  GL_RHS_r_elec(i,j,k); 
               }
            });
        }
}


void Calculate_Landau(Array<MultiFab, AMREX_SPACEDIM> &GL_rhs_Landau,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                MultiFab&                       Gamma,
                MultiFab&                 tphaseMask)
{
        // loop over boxes
        for ( MFIter mfi(P_old[0]); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real> &GL_RHS_p = GL_rhs_Landau[0].array(mfi);
            const Array4<Real> &GL_RHS_q = GL_rhs_Landau[1].array(mfi);
            const Array4<Real> &GL_RHS_r = GL_rhs_Landau[2].array(mfi);
            const Array4<Real> &pOld_p = P_old[0].array(mfi);
            const Array4<Real> &pOld_q = P_old[1].array(mfi);
            const Array4<Real> &pOld_r = P_old[2].array(mfi);
            const Array4<Real>& Gam = Gamma.array(mfi);
            const Array4<Real>& tphase = tphaseMask.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real dFdPp_Landau = alpha*pOld_p(i,j,k) + beta*std::pow(pOld_p(i,j,k),3.) + FerroX::gamma*std::pow(pOld_p(i,j,k),5.)
                                    + 2. * alpha_12 * pOld_p(i,j,k) * std::pow(pOld_q(i,j,k),2.)
                                    + 2. * alpha_12 * pOld_p(i,j,k) * std::pow(pOld_r(i,j,k),2.)
                                    + 4. * alpha_112 * std::pow(pOld_p(i,j,k),3.) * (std::pow(pOld_q(i,j,k),2.) + std::pow(pOld_r(i,j,k),2.))
                                    + 2. * alpha_112 * pOld_p(i,j,k) * std::pow(pOld_q(i,j,k),4.)
                                    + 2. * alpha_112 * pOld_p(i,j,k) * std::pow(pOld_r(i,j,k),4.)
                                    + 2. * alpha_123 * pOld_p(i,j,k) * std::pow(pOld_q(i,j,k),2.) * std::pow(pOld_r(i,j,k),2.);

                Real dFdPq_Landau = alpha*pOld_q(i,j,k) + beta*std::pow(pOld_q(i,j,k),3.) + FerroX::gamma*std::pow(pOld_q(i,j,k),5.)
                                    + 2. * alpha_12 * pOld_q(i,j,k) * std::pow(pOld_p(i,j,k),2.)
                                    + 2. * alpha_12 * pOld_q(i,j,k) * std::pow(pOld_r(i,j,k),2.)
                                    + 4. * alpha_112 * std::pow(pOld_q(i,j,k),3.) * (std::pow(pOld_p(i,j,k),2.) + std::pow(pOld_r(i,j,k),2.))
                                    + 2. * alpha_112 * pOld_q(i,j,k) * std::pow(pOld_p(i,j,k),4.)
                                    + 2. * alpha_112 * pOld_q(i,j,k) * std::pow(pOld_r(i,j,k),4.)
                                    + 2. * alpha_123 * pOld_q(i,j,k) * std::pow(pOld_p(i,j,k),2.) * std::pow(pOld_r(i,j,k),2.);
                
                Real dFdPr_Landau = alpha*pOld_r(i,j,k) + beta*std::pow(pOld_r(i,j,k),3.) + FerroX::gamma*std::pow(pOld_r(i,j,k),5.)
                                    + 2. * alpha_12 * pOld_r(i,j,k) * std::pow(pOld_p(i,j,k),2.)
                                    + 2. * alpha_12 * pOld_r(i,j,k) * std::pow(pOld_q(i,j,k),2.)
                                    + 4. * alpha_112 * std::pow(pOld_r(i,j,k),3.) * (std::pow(pOld_p(i,j,k),2.) + std::pow(pOld_q(i,j,k),2.))
                                    + 2. * alpha_112 * pOld_r(i,j,k) * std::pow(pOld_p(i,j,k),4.)
                                    + 2. * alpha_112 * pOld_r(i,j,k) * std::pow(pOld_q(i,j,k),4.)
                                    + 2. * alpha_123 * pOld_r(i,j,k) * std::pow(pOld_p(i,j,k),2.) * std::pow(pOld_q(i,j,k),2.);

                GL_RHS_p(i,j,k) = -1.0 * Gam(i,j,k) * dFdPp_Landau;
                GL_RHS_q(i,j,k) = -1.0 * Gam(i,j,k) * dFdPq_Landau;
                GL_RHS_r(i,j,k) = -1.0 * Gam(i,j,k) * dFdPr_Landau;

                if (is_polarization_scalar == 1){
		   GL_RHS_p(i,j,k) = 0.0;
		   GL_RHS_q(i,j,k) = 0.0;
		}

		//set t_phase GL_RHS_r to zero so that it stays zero. It is initialized to zero in t-phase as well
                //if(x <= t_phase_hi[0] && x >= t_phase_lo[0] && y <= t_phase_hi[1] && y >= t_phase_lo[1] && z <= t_phase_hi[2] && z >= t_phase_lo[2]){
                if (tphase(i,j,k) == 1.0){
		   GL_RHS_p(i,j,k) = 0.0;
		   GL_RHS_q(i,j,k) = 0.0;
		   GL_RHS_r(i,j,k) = 0.0;
                }
            });
        }
}


void Calculate_Grad(Array<MultiFab, AMREX_SPACEDIM> &GL_rhs_grad,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                MultiFab&                       Gamma,
                MultiFab&                 MaterialMask,
                MultiFab&                 tphaseMask,
                MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta,
                const Geometry& geom)
{
        // loop over boxes
        for ( MFIter mfi(P_old[0]); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real> &GL_RHS_p = GL_rhs_grad[0].array(mfi);
            const Array4<Real> &GL_RHS_q = GL_rhs_grad[1].array(mfi);
            const Array4<Real> &GL_RHS_r = GL_rhs_grad[2].array(mfi);
            const Array4<Real> &pOld_p = P_old[0].array(mfi);
            const Array4<Real> &pOld_q = P_old[1].array(mfi);
            const Array4<Real> &pOld_r = P_old[2].array(mfi);
            const Array4<Real>& Gam = Gamma.array(mfi);
            const Array4<Real>& mask = MaterialMask.array(mfi);
            const Array4<Real>& tphase = tphaseMask.array(mfi);

            const Array4<Real> &angle_alpha_arr = angle_alpha.array(mfi);
            const Array4<Real> &angle_beta_arr = angle_beta.array(mfi);
            const Array4<Real> &angle_theta_arr = angle_theta.array(mfi);


            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {

               //Convert Euler angles from degrees to radians
               amrex::Real Pi = 3.14159265358979323846; 
               amrex::Real alpha_rad = Pi/180.*angle_alpha_arr(i,j,k);
               amrex::Real beta_rad =  Pi/180.*angle_beta_arr(i,j,k);
               amrex::Real theta_rad = Pi/180.*angle_theta_arr(i,j,k);
  
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

                Real dFdPp_grad = - g11 * DoubleDPDx(pOld_p, mask, i, j, k, dx)
                                  - (g44 + g44_p) * DoubleDPDy(pOld_p, mask, i, j, k, dx)
                                  - (g44 + g44_p) * DoubleDPDz(pOld_p, mask, i, j, k, dx)
                                  - (g12 + g44 - g44_p) * DoubleDPDxDy(pOld_q, mask, i, j, k, dx)  // d2P/dxdy
                                  - (g12 + g44 - g44_p) * DoubleDPDxDz(pOld_r, mask, i, j, k, dx); // d2P/dxdz
                
                Real dFdPq_grad = - g11 * DoubleDPDy(pOld_q, mask, i, j, k, dx)
                                  - (g44 - g44_p) * DoubleDPDx(pOld_q, mask, i, j, k, dx)
                                  - (g44 - g44_p) * DoubleDPDz(pOld_q, mask, i, j, k, dx)
                                  - (g12 + g44 + g44_p) * DoubleDPDxDy(pOld_p, mask, i, j, k, dx) // d2P/dxdy
                                  - (g12 + g44 - g44_p) * DoubleDPDyDz(pOld_r, mask, i, j, k, dx);// d2P/dydz

                Real dFdPr_grad = - g11 * ( R_31*R_31*DoubleDPDx(pOld_r, mask, i, j, k, dx)
                                           +R_32*R_32*DoubleDPDy(pOld_r, mask, i, j, k, dx)
                                           +R_33*R_33*DoubleDPDz(pOld_r, mask, i, j, k, dx)
                                           +2.*R_31*R_32*DoubleDPDxDy(pOld_r, mask, i, j, k, dx)
                                           +2.*R_32*R_33*DoubleDPDyDz(pOld_r, mask, i, j, k, dx)
                                           +2.*R_33*R_31*DoubleDPDxDz(pOld_r, mask, i, j, k, dx))
                                           
                                  - (g44 - g44_p) * ( R_11*R_11*DoubleDPDx(pOld_r, mask, i, j, k, dx) 
                                                     +R_12*R_12*DoubleDPDy(pOld_r, mask, i, j, k, dx) 
                                                     +R_13*R_13*DoubleDPDz(pOld_r, mask, i, j, k, dx) 
                                                     +2.*R_11*R_12*DoubleDPDxDy(pOld_r, mask, i, j, k, dx) 
                                                     +2.*R_12*R_13*DoubleDPDyDz(pOld_r, mask, i, j, k, dx) 
                                                     +2.*R_13*R_11*DoubleDPDxDz(pOld_r, mask, i, j, k, dx))

                                  - (g44 - g44_p) * ( R_21*R_21*DoubleDPDx(pOld_r, mask, i, j, k, dx)
                                                     +R_22*R_22*DoubleDPDy(pOld_r, mask, i, j, k, dx)
                                                     +R_23*R_23*DoubleDPDz(pOld_r, mask, i, j, k, dx)
                                                     +2.*R_21*R_22*DoubleDPDxDy(pOld_r, mask, i, j, k, dx)
                                                     +2.*R_22*R_23*DoubleDPDyDz(pOld_r, mask, i, j, k, dx)
                                                     +2.*R_23*R_21*DoubleDPDxDz(pOld_r, mask, i, j, k, dx))

                                  - (g44 + g44_p + g12) * DoubleDPDyDz(pOld_q, mask, i, j, k, dx) // d2P/dydz
                                  - (g44 + g44_p + g12) * DoubleDPDxDz(pOld_p, mask, i, j, k, dx); // d2P/dxdz

                GL_RHS_p(i,j,k) = -1.0 * Gam(i,j,k) * dFdPp_grad;
                GL_RHS_q(i,j,k) = -1.0 * Gam(i,j,k) * dFdPq_grad;
                GL_RHS_r(i,j,k) = -1.0 * Gam(i,j,k) * dFdPr_grad;

                if (is_polarization_scalar == 1){
		   GL_RHS_p(i,j,k) = 0.0;
		   GL_RHS_q(i,j,k) = 0.0;
		}

		//set t_phase GL_RHS_r to zero so that it stays zero. It is initialized to zero in t-phase as well
                //if(x <= t_phase_hi[0] && x >= t_phase_lo[0] && y <= t_phase_hi[1] && y >= t_phase_lo[1] && z <= t_phase_hi[2] && z >= t_phase_lo[2]){
                if (tphase(i,j,k) == 1.0){
		   GL_RHS_p(i,j,k) = 0.0;
		   GL_RHS_q(i,j,k) = 0.0;
		   GL_RHS_r(i,j,k) = 0.0;
                }
            });
        }
}


void Calculate_Elec(Array<MultiFab, AMREX_SPACEDIM> &GL_rhs_elec,
                Array<MultiFab, AMREX_SPACEDIM> &E,
                MultiFab&                       Gamma,
                MultiFab&                 tphaseMask)
{
        // loop over boxes
        for ( MFIter mfi(E[0]); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            const Array4<Real> &GL_RHS_p = GL_rhs_elec[0].array(mfi);
            const Array4<Real> &GL_RHS_q = GL_rhs_elec[1].array(mfi);
            const Array4<Real> &GL_RHS_r = GL_rhs_elec[2].array(mfi);
            const Array4<Real> &Ep = E[0].array(mfi);
            const Array4<Real> &Eq = E[1].array(mfi);
            const Array4<Real> &Er = E[2].array(mfi);
            const Array4<Real>& Gam = Gamma.array(mfi);
            const Array4<Real>& tphase = tphaseMask.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {

                GL_RHS_p(i,j,k) = -1.0 * Gam(i,j,k) * (-Ep(i,j,k));
                GL_RHS_q(i,j,k) = -1.0 * Gam(i,j,k) * (-Eq(i,j,k));
                GL_RHS_r(i,j,k) = -1.0 * Gam(i,j,k) * (-Er(i,j,k));

                if (is_polarization_scalar == 1){
		   GL_RHS_p(i,j,k) = 0.0;
		   GL_RHS_q(i,j,k) = 0.0;
		}

		//set t_phase GL_RHS_r to zero so that it stays zero. It is initialized to zero in t-phase as well
                //if(x <= t_phase_hi[0] && x >= t_phase_lo[0] && y <= t_phase_hi[1] && y >= t_phase_lo[1] && z <= t_phase_hi[2] && z >= t_phase_lo[2]){
                if (tphase(i,j,k) == 1.0){
		   GL_RHS_p(i,j,k) = 0.0;
		   GL_RHS_q(i,j,k) = 0.0;
		   GL_RHS_r(i,j,k) = 0.0;
                }
            });
        }
}


