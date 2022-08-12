#include "TotalEnergyDensity.H"

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
                Real                            Phi_Bc_lo,
                Real                            Phi_Bc_hi,
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
                    Real P_int = ( 9.*pOld(i,j,k) - pOld(i,j,k+1) ) / (8. + 3.*dx[2] / lambda);
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
		    Real P_int = ( 9.*pOld(i,j,k) - pOld(i,j,k-1) ) / (8. - 3.*dx[2] / lambda);
                    Real dPdz = P_int/lambda;
                    d2P_z = (dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k-1))/dx[2]/dx[2];//2nd Order
                  } else if (P_BC_flag_hi == 2){
                    Real dPdz = 0.;
                    d2P_z = (dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k-1))/dx[2]/dx[2];//2nd Order
                  }

                  grad_term = g11 * d2P_z;
                  phi_term = (4.*Phi_Bc_hi - 3.*phi(i,j,k) - phi(i,j,k-1))/(3.*dx[2]);

                } else{ //inside FE

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


