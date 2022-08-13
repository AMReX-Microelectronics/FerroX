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
                Real z    = prob_lo[2] + (k+0.5) * dx[2];
                Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

                GL_RHS(i,j,k)  = -1.0 * Gam(i,j,k) *
                    (  alpha*pOld(i,j,k) + beta*std::pow(pOld(i,j,k),3.) + gamma*std::pow(pOld(i,j,k),5.)
                     - g44 * (pOld(i+1,j,k) - 2.*pOld(i,j,k) + pOld(i-1,j,k)) / (dx[0]*dx[0])
                     - g44 * (pOld(i,j+1,k) - 2.*pOld(i,j,k) + pOld(i,j-1,k)) / (dx[1]*dx[1])
                     - g11 * DoubleDPDz(pOld, z, z_hi, z_lo, P_BC_flag_lo, P_BC_flag_hi, FE_lo, FE_hi, lambda, i, j, k, dx)
                     + DPhiDz(phi, z, z_hi, z_lo, prob_lo, prob_hi, Phi_Bc_hi, Phi_Bc_lo, i, j, k, dx)
                    );
            });
        }
}
