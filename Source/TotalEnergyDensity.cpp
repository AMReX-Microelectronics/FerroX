#include "TotalEnergyDensity.H"
#include "DerivativeAlgorithm.H"

void CalculateTDGL_RHS(Array<MultiFab, AMREX_SPACEDIM> &GL_rhs,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                MultiFab&                       PoissonPhi,
                MultiFab&                       Gamma,
                const Geometry& geom)
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
            const Array4<Real>& phi = PoissonPhi.array(mfi);
            const Array4<Real>& Gam = Gamma.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                Real grad_term, phi_term, d2P_z;
                Real z    = prob_lo[2] + (k+0.5) * dx[2];
                Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

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

                Real dFdPx_grad = - g11 * DoubleDPDx(pOld_x, i, j, k, dx)
                                  - (g44 + g44_p) * DoubleDPDy(pOld_x, i, j, k, dx)
                                  - (g44 + g44_p) * DoubleDPDz(pOld_x, z, z_hi, z_lo, i, j, k, dx)
                                  - (g12 + g44 - g44_p) * (DFDy(pOld_y, i+1, j, k, dx) - DFDy(pOld_y, i-1, j, k, dx)) / 2. /dx[0];
                                  - (g12 + g44 - g44_p) * (DPDz(pOld_z, z, z_hi, z_lo, i+1, j, k, dx)
                                                         - DPDz(pOld_z, z, z_hi, z_lo, i-1, j, k, dx)) / 2. /dx[0]; // d2P/dxdz
                
                Real dFdPy_grad = - g11 * DoubleDPDy(pOld_y, i, j, k, dx)
                                  - (g44 - g44_p) * DoubleDPDx(pOld_y, i, j, k, dx)
                                  - (g44 - g44_p) * DoubleDPDz(pOld_y, z, z_hi, z_lo, i, j, k, dx)
                                  - (g12 + g44 + g44_p) * (DFDx(pOld_x, i, j+1, k, dx) - DFDx(pOld_x, i, j-1, k, dx)) / 2. / dx[1]
                                  - (g12 + g44 - g44_p) * (DPDz(pOld_z, z, z_hi, z_lo, i, j+1, k, dx) 
                                                         - DPDz(pOld_z, z, z_hi, z_lo, i, j-1, k, dx)) / 2. /dx[1];

                Real dFdPz_grad = - g11 * DoubleDPDz(pOld_z, z, z_hi, z_lo, i, j, k, dx)
                                  - (g44 - g44_p) * DoubleDPDx(pOld_z, i, j, k, dx)
                                  - (g44 - g44_p) * DoubleDPDy(pOld_z, i, j, k, dx)
                                  - (g44 + g44_p + g12) * (DPDz(pOld_y, z, z_hi, z_lo, i, j+1, k, dx) 
                                                         - DPDz(pOld_y, z, z_hi, z_lo, i, j-1, k, dx)) / 2. /dx[1]; // d2P/dydz
                                  - (g44 + g44_p + g12) * (DPDz(pOld_x, z, z_hi, z_lo, i+1, j, k, dx) 
                                                         - DPDz(pOld_x, z, z_hi, z_lo, i-1, j, k, dx)) / 2. /dx[0]; // d2P/dxdz

                GL_RHS_x(i,j,k)  = -1.0 * Gam(i,j,k) *
                    (  dFdPx_Landau
                     + dFdPx_grad
                     + DFDx(phi, i, j, k, dx)
                    );

                GL_RHS_y(i,j,k)  = -1.0 * Gam(i,j,k) *
                    (  dFdPy_Landau
                     + dFdPy_grad
                     + DFDy(phi, i, j, k, dx)
                    );

                GL_RHS_z(i,j,k)  = -1.0 * Gam(i,j,k) *
                    (  dFdPz_Landau
                     + dFdPz_grad
                     + DphiDz(phi, z, z_hi, z_lo, i, j, k, dx)
                    );

            });
        }
}


