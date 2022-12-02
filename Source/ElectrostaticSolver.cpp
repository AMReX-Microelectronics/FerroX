#include "ElectrostaticSolver.H"
#include "DerivativeAlgorithm.H"
#include "ChargeDensity.H"
#include "Utils/CodeUtils/CodeUtil.H"

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
                 Real z    = prob_lo[2] + (k+0.5) * dx[2];
                 Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                 Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

                 if(z <= SC_hi[2]){ //SC region

                   RHS(i,j,k) = charge_den_arr(i,j,k);

                 } else if(z < DE_hi[2]){ //DE region

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
        PoissonPhi_plus_delta.plus(delta, 0, 1, 0); 

        // Calculate rho from Phi in SC region
        ComputeRho(PoissonPhi_plus_delta, rho, e_den, p_den, geom);

        //Compute RHS of Poisson equation
        ComputePoissonRHS(PoissonRHS_phi_plus_delta, P_old, rho, geom);

        MultiFab::LinComb(alpha_cc, 1./delta, PoissonRHS_phi_plus_delta, 0, -1./delta, PoissonRHS, 0, 0, 1, 0);
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
                     Real z    = prob_lo[2] + (k+0.5) * dx[2];
                     Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                     Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

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
          Real z = prob_lo[2] + (k+0.5) * dx[2];
          if(z < SC_hi[2]-small) {
             beta_f0(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  } else if(z >= SC_hi[2]-small && z < SC_hi[2]+small){
             beta_f0(i,j,k) = 0.5*(epsilon_de + epsilon_si) * epsilon_0; //SC-DE interface
          } else if(z < DE_hi[2]-small) {
             beta_f0(i,j,k) = epsilon_de * epsilon_0; //DE layer
	  } else if(z >= DE_hi[2]-small && z < DE_hi[2]+small){
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
          if(z < SC_hi[2]-small) {
             beta_f1(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  } else if(z >= SC_hi[2]-small && z < SC_hi[2]+small){
             beta_f1(i,j,k) = 0.5*(epsilon_de + epsilon_si) * epsilon_0; //SC-DE interface
          } else if(z < DE_hi[2]-small) {
             beta_f1(i,j,k) = epsilon_de * epsilon_0; //DE layer
	  } else if(z >= DE_hi[2]-small && z < DE_hi[2]+small){
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
          if(z < SC_hi[2]-small) {
             beta_f2(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  } else if(z >= SC_hi[2]-small && z < SC_hi[2]+small && SC_hi[2] > prob_lo[2]+small){
             beta_f2(i,j,k) = 0.5*(epsilon_de + epsilon_si) * epsilon_0; //SC-DE interface
          } else if(z < DE_hi[2]-small) {
             beta_f2(i,j,k) = epsilon_de * epsilon_0; //DE layer
	  } else if(z >= DE_hi[2]-small && z < DE_hi[2]+small){
             beta_f2(i,j,k) = 0.5*(epsilon_de + epsilonZ_fe) * epsilon_0; //DE-FE interface
             //beta_f2(i,j,k) = epsilon_de * epsilon_0; //DE-FE interface
          } else {
             beta_f2(i,j,k) = epsilonZ_fe * epsilon_0; //FE layer
          }
        });
    }

}

void SetPoissonBC(c_Code& rCode, std::array<std::array<amrex::LinOpBCType,AMREX_SPACEDIM>,2> LinOpBCType_2d, bool all_homogeneous_boundaries, bool some_functionbased_inhomogeneous_boundaries, bool some_constant_inhomogeneous_boundaries)
{
    auto& rBC = rCode.get_BoundaryConditions();
    auto& map_boundary_type = rBC.map_boundary_type;
    auto& bcType_2d = rBC.bcType_2d;
    auto& map_bcAny_2d = rBC.map_bcAny_2d;

    for (std::size_t i = 0; i < 2; ++i)
    {
        for (std::size_t j = 0; j < AMREX_SPACEDIM; ++j)
        {

            switch(map_boundary_type[ bcType_2d[i][j] ])
            {
                case s_BoundaryConditions::dir :
                {
                     LinOpBCType_2d[i][j] = LinOpBCType::Dirichlet;

                     if(map_bcAny_2d[i][j] == "inhomogeneous_constant")
                     {
                         all_homogeneous_boundaries = false;
                         some_constant_inhomogeneous_boundaries = true;
                     }
                     if(map_bcAny_2d[i][j] == "inhomogeneous_function")
                     {
                         all_homogeneous_boundaries = false;
                         some_functionbased_inhomogeneous_boundaries = true;
                     }
                     break;
                }
                case s_BoundaryConditions::neu :
                {
                     if(map_bcAny_2d[i][j] == "homogeneous")
                     {
                         LinOpBCType_2d[i][j] = LinOpBCType::Neumann;
                     }
                     else if(map_bcAny_2d[i][j] == "inhomogeneous_constant")
                     {
                         LinOpBCType_2d[i][j] = LinOpBCType::inhomogNeumann;
                         all_homogeneous_boundaries = false;
                         some_constant_inhomogeneous_boundaries = true;
                     }
                     else if(map_bcAny_2d[i][j] == "inhomogeneous_function")
                     {
                         LinOpBCType_2d[i][j] = LinOpBCType::inhomogNeumann;
                         all_homogeneous_boundaries = false;
                         some_functionbased_inhomogeneous_boundaries = true;
                     }
                     break;
                }
                case s_BoundaryConditions::per :
                {
                    LinOpBCType_2d[i][j] = LinOpBCType::Periodic;
                    break;
                }
            }

        }
    }

}

void Fill_Constant_Inhomogeneous_Boundaries(c_Code& rCode, MultiFab& PoissonPhi)
{
    auto& rGprop = rCode.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    auto& rBC = rCode.get_BoundaryConditions();
    auto& bcAny_2d = rBC.bcAny_2d;
    auto& map_bcAny_2d = rBC.map_bcAny_2d;

    std::vector<int> dir_inhomo_const_lo;
    std::string value = "inhomogeneous_constant";
    bool found_lo = findByValue(dir_inhomo_const_lo, map_bcAny_2d[0], value);
    std::vector<int> dir_inhomo_const_hi;
    bool found_hi = findByValue(dir_inhomo_const_hi, map_bcAny_2d[1], value);

    int len = 1;
    for (MFIter mfi(PoissonPhi, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& phi_arr = PoissonPhi.array(mfi);

        const auto& bx = mfi.tilebox();
        
        if(found_lo) {
            for (auto dir : dir_inhomo_const_lo) 
	    {
                if (bx.smallEnd(dir) == domain.smallEnd(dir)) 
		{
	            auto value = std::any_cast<amrex::Real>(bcAny_2d[0][dir]);		
                    Box const& bxlo = amrex::adjCellLo(bx, dir,len);
                    amrex::ParallelFor(bxlo,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        phi_arr(i,j,k) = value;
                    });
                }
            }
        }
        if(found_hi) {
            for (auto dir : dir_inhomo_const_hi) 
	    {
                if (bx.bigEnd(dir) == domain.bigEnd(dir)) 
		{
		    auto value = std::any_cast<amrex::Real>(bcAny_2d[1][dir]);	
                    Box const& bxhi = amrex::adjCellHi(bx, dir,len);
                    amrex::ParallelFor(bxhi,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        phi_arr(i,j,k) = value;
                    });
                }
            }
        }
    } 

}
void Fill_FunctionBased_Inhomogeneous_Boundaries(c_Code& rCode, MultiFab& PoissonPhi)
{
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
            Phi(i,j,k) = Phi_Bc_lo;
          } else if(k >= n_cell[2]){
            Phi(i,j,k) = Phi_Bc_hi;
          }
        });
    }
}
