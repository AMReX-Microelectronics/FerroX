#include "ElectrostaticSolver.H"
#include "DerivativeAlgorithm.H"
#include "ChargeDensity.H"
#include "Utils/eXstaticUtils/eXstaticUtil.H"

void ComputePoissonRHS(MultiFab&               PoissonRHS,
                Array<MultiFab, AMREX_SPACEDIM> &P_old,
                MultiFab&                       rho,
                MultiFab&                 MaterialMask,
                MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta,
                const Geometry&                 geom)
{
    for ( MFIter mfi(PoissonRHS); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();
            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real> &pOld_p = P_old[0].array(mfi);
            const Array4<Real> &pOld_q = P_old[1].array(mfi);
            const Array4<Real> &pOld_r = P_old[2].array(mfi);
            const Array4<Real>& RHS = PoissonRHS.array(mfi);
            const Array4<Real>& charge_den_arr = rho.array(mfi);
            const Array4<Real>& mask = MaterialMask.array(mfi);

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

                 if(mask(i,j,k) >= 2.0){ //SC region

                   RHS(i,j,k) = charge_den_arr(i,j,k);

                 } else if(mask(i,j,k) == 1.0){ //DE region

                   RHS(i,j,k) = 0.;

                 } else { //mask(i,j,k) == 0.0 FE region
                   RHS(i,j,k) = - (R_11*DPDx(pOld_p, mask, i, j, k, dx) + R_12*DPDy(pOld_p, mask, i, j, k, dx) + R_13*DPDz(pOld_p, mask, i, j, k, dx))
                                - (R_21*DPDx(pOld_q, mask, i, j, k, dx) + R_22*DPDy(pOld_q, mask, i, j, k, dx) + R_23*DPDz(pOld_q, mask, i, j, k, dx))
                                - (R_31*DPDx(pOld_r, mask, i, j, k, dx) + R_32*DPDy(pOld_r, mask, i, j, k, dx) + R_33*DPDz(pOld_r, mask, i, j, k, dx));

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
	     MultiFab&            MaterialMask,
             MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta,
             const          Geometry& geom,
	     const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
             const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi)

{
   
        MultiFab PoissonPhi_plus_delta(PoissonPhi.boxArray(), PoissonPhi.DistributionMap(), 1, 0); 
        MultiFab PoissonRHS_phi_plus_delta(PoissonRHS.boxArray(), PoissonRHS.DistributionMap(), 1, 0); 
 
        MultiFab::Copy(PoissonPhi_plus_delta, PoissonPhi, 0, 0, 1, 0); 
        PoissonPhi_plus_delta.plus(delta, 0, 1, 0); 

        // Calculate rho from Phi in SC region
        ComputeRho(PoissonPhi, rho, e_den, p_den, MaterialMask);

        //Compute RHS of Poisson equation
        ComputePoissonRHS(PoissonRHS_phi_plus_delta, P_old, rho, MaterialMask, angle_alpha, angle_beta, angle_theta, geom);

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
                Array<MultiFab, AMREX_SPACEDIM>& E,
                MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta,
                const Geometry&                 geom,
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo, 
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi)
{
       // Calculate E from Phi

        for ( MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();

            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real>& Ep_arr = E[0].array(mfi);
            const Array4<Real>& Eq_arr = E[1].array(mfi);
            const Array4<Real>& Er_arr = E[2].array(mfi);
            const Array4<Real>& phi = PoissonPhi.array(mfi);

            const Array4<Real> &angle_alpha_arr = angle_alpha.array(mfi);
            const Array4<Real> &angle_beta_arr = angle_beta.array(mfi);
            const Array4<Real> &angle_theta_arr = angle_theta.array(mfi);


            amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                     Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                     Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

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

                     Ep_arr(i,j,k) = - (R_11*DFDx(phi, i, j, k, dx) + R_12*DFDy(phi, i, j, k, dx) + R_13*DphiDz(phi, z_hi, z_lo, i, j, k, dx, prob_lo, prob_hi));
                     Eq_arr(i,j,k) = - (R_21*DFDx(phi, i, j, k, dx) + R_22*DFDy(phi, i, j, k, dx) + R_23*DphiDz(phi, z_hi, z_lo, i, j, k, dx, prob_lo, prob_hi));
                     Er_arr(i,j,k) = - (R_31*DFDx(phi, i, j, k, dx) + R_32*DFDy(phi, i, j, k, dx) + R_33*DphiDz(phi, z_hi, z_lo, i, j, k, dx, prob_lo, prob_hi));


             });
        }

}

void InitializePermittivity(std::array<std::array<amrex::LinOpBCType,AMREX_SPACEDIM>,2>& LinOpBCType_2d, 
		MultiFab& beta_cc,
	       	const MultiFab& MaterialMask,
	       	const MultiFab& tphaseMask,
	       	const amrex::GpuArray<int, AMREX_SPACEDIM>& n_cell,
	       	const Geometry& geom, 
		const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
	       	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi)
{

    beta_cc.setVal(0.0);

    // set cell-centered beta coefficient to
    // epsilon values in SC, FE, and DE layers
    // loop over boxes
    for (MFIter mfi(beta_cc); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        const Array4<Real>& beta = beta_cc.array(mfi);
        const Array4<Real const>& mask = MaterialMask.array(mfi);
        const Array4<Real const>& tphase = tphaseMask.array(mfi);

	// extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {

	  Real x = prob_lo[0] + (i+0.5) * dx[0];
	  Real y = prob_lo[1] + (j+0.5) * dx[1];
	  Real z = prob_lo[1] + (k+0.5) * dx[2];
	
          if(mask(i,j,k) == 0.0) {
             beta(i,j,k) = epsilonX_fe * epsilon_0; //FE layer
	     //set t_phase beta to epsilonX_fe_tphase
	     //if(x <= t_phase_hi[0] && x >= t_phase_lo[0] && y <= t_phase_hi[1] && y >= t_phase_lo[1] && z <= t_phase_hi[2] && z >= t_phase_lo[2]){
	     if(tphase(i,j,k) == 1.0){
               beta(i,j,k) = epsilonX_fe_tphase * epsilon_0;
             }
          } else if(mask(i,j,k) == 1.0) {
             beta(i,j,k) = epsilon_de * epsilon_0; //DE layer
          } else if (mask(i,j,k) >= 2.0){
             beta(i,j,k) = epsilon_si * epsilon_0; //SC layer
          } else {
             beta(i,j,k) = epsilon_de * epsilon_0; //Spacer is same as DE
	  }

        });
    }
    beta_cc.FillBoundary(geom.periodicity()); //For Periodic BC and internal domain decomposition

    //For Non-periodic Poisson BC, fill the ghost cells with the value in the adjacent cell in valid domain
    for (MFIter mfi(beta_cc); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(1);

        const Array4<Real>& beta = beta_cc.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k)
        {
	        if (LinOpBCType_2d[0][0] == amrex::LinOpBCType::Dirichlet || LinOpBCType_2d[0][0] == amrex::LinOpBCType::Neumann ){
           	   if(i < 0) {
		     beta(i,j,k) = beta(i+1,j,k);
		   }
		}

	        if (LinOpBCType_2d[1][0] == amrex::LinOpBCType::Dirichlet || LinOpBCType_2d[1][0] == amrex::LinOpBCType::Neumann ){
		   if(i > n_cell[0] - 1) {
		     beta(i,j,k) = beta(i-1,j,k);
		   }
		}
	        if (LinOpBCType_2d[0][1] == amrex::LinOpBCType::Dirichlet || LinOpBCType_2d[0][1] == amrex::LinOpBCType::Neumann ){
  		   if(j < 0) {
		     beta(i,j,k) = beta(i,j+1,k);
		   }
		}
	        if (LinOpBCType_2d[1][1] == amrex::LinOpBCType::Dirichlet || LinOpBCType_2d[1][1] == amrex::LinOpBCType::Neumann ){
		   if(j > n_cell[1] - 1) {
		     beta(i,j,k) = beta(i,j-1,k);
		   }
		}
	        if (LinOpBCType_2d[0][2] == amrex::LinOpBCType::Dirichlet || LinOpBCType_2d[0][2] == amrex::LinOpBCType::Neumann ){
  		   if(k < 0) {
		     beta(i,j,k) = beta(i,j,k+1);
		   }
		}
	        if (LinOpBCType_2d[1][2] == amrex::LinOpBCType::Dirichlet || LinOpBCType_2d[1][2] == amrex::LinOpBCType::Neumann ){
		   if(k > n_cell[2] - 1) {
		     beta(i,j,k) = beta(i,j,k-1);
		   }
		}
        });
    }
}

void SetPoissonBC(c_FerroX& rFerroX, std::array<std::array<amrex::LinOpBCType,AMREX_SPACEDIM>,2>& LinOpBCType_2d, bool& all_homogeneous_boundaries, bool& some_functionbased_inhomogeneous_boundaries, bool& some_constant_inhomogeneous_boundaries)
{
    auto& rBC = rFerroX.get_BoundaryConditions();
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

void Fill_Constant_Inhomogeneous_Boundaries(c_FerroX& rFerroX, MultiFab& PoissonPhi)
{
    auto& rGprop = rFerroX.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    auto& rBC = rFerroX.get_BoundaryConditions();
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
void Fill_FunctionBased_Inhomogeneous_Boundaries(c_FerroX& rFerroX, MultiFab& PoissonPhi, amrex::Real& time)
{
    auto& rGprop = rFerroX.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv = PoissonPhi.ixType().toIntVect();

    auto& rBC = rFerroX.get_BoundaryConditions();
    auto& bcAny_2d = rBC.bcAny_2d;
    auto& map_bcAny_2d = rBC.map_bcAny_2d;

    std::vector<int> dir_inhomo_func_lo;
    std::string value = "inhomogeneous_function";
    bool found_lo = findByValue(dir_inhomo_func_lo, map_bcAny_2d[0], value);
    std::vector<int> dir_inhomo_func_hi;
    bool found_hi = findByValue(dir_inhomo_func_hi, map_bcAny_2d[1], value);

    for (MFIter mfi(PoissonPhi, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& soln_arr = PoissonPhi.array(mfi);
        const auto& bx = mfi.tilebox();
        
        /*for low sides*/
        if(found_lo)
        {
            for (auto dir : dir_inhomo_func_lo) //looping over boundaries of type inhomogeneous_function
            {
                if (bx.smallEnd(dir) == domain.smallEnd(dir)) //work with a box that adjacent to the domain boundary
                { 
                    Box const& bxlo = amrex::adjCellLo(bx, dir);
                    std::string macro_str = std::any_cast<std::string>(bcAny_2d[0][dir]);

                    auto pParser = rBC.get_p_parser(macro_str);
		    #ifdef TIME_DEPENDENT
		        const auto& macro_parser = pParser->compile<4>();
		    #else
		        const auto& macro_parser = pParser->compile<3>();
		    #endif

                    amrex::ParallelFor(bxlo,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
		        #ifdef TIME_DEPENDENT
                            eXstatic_MFab_Util::ConvertParserIntoMultiFab_4vars(i,j,k,time,dx,real_box,iv,macro_parser,soln_arr);  
		        #else
                            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv,macro_parser,soln_arr);  
                        #endif
                    });
                }
            }
        }
	if(found_hi)
        {
            for (auto dir : dir_inhomo_func_hi) //looping over boundaries of type inhomogeneous_function
            {
                if (bx.bigEnd(dir) == domain.bigEnd(dir)) //work with a box that adjacent to the domain boundary
                {
                    Box const& bxhi = amrex::adjCellHi(bx, dir);
		    std::string macro_str = std::any_cast<std::string>(bcAny_2d[1][dir]);

                        auto pParser = rBC.get_p_parser(macro_str);
			#ifdef TIME_DEPENDENT
			    const auto& macro_parser = pParser->compile<4>();
			#else
			    const auto& macro_parser = pParser->compile<3>();
			#endif

                        amrex::ParallelFor(bxhi,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
		            #ifdef TIME_DEPENDENT
                                eXstatic_MFab_Util::ConvertParserIntoMultiFab_4vars(i,j,k,time,dx,real_box,iv,macro_parser,soln_arr);
		            #else
                                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv,macro_parser,soln_arr);
                            #endif
                        });
	        }
            }
        }

    }
}

void SetPhiBC_z(MultiFab& PoissonPhi, const amrex::GpuArray<int, AMREX_SPACEDIM>& n_cell)
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

