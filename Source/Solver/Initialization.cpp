#include "Initialization.H"
#include "Utils/eXstaticUtils/eXstaticUtil.H"
#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"

// INITIALIZE rho in SC region
void InitializePandRho(Array<MultiFab, AMREX_SPACEDIM> &P_old,
                   MultiFab&   BigGamma,
                   MultiFab&   rho,
                   MultiFab&   e_den,
                   MultiFab&   p_den,
		           const MultiFab& MaterialMask,
		           const MultiFab& tphaseMask,
                   const amrex::GpuArray<int, AMREX_SPACEDIM>& n_cell,
                   const       Geometry& geom,
		           const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                   const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi)
{

    if (prob_type == 1) {  //2D : Initialize uniform P in y direction

       amrex::Print() << "==================================""\n"
                         "P is initialized for a 2D problem." "\n"
                         "==================================""\n" ;

    } else if (prob_type == 2) { // 3D : Initialize random P

       amrex::Print() << "==================================""\n"
                         "P is initialized for a 3D problem." "\n"
                         "==================================""\n" ;

    } else if (prob_type == 3) {

       amrex::Print() << "==================================""\n"
                         "P is initialized for convergence test." "\n"
                         "==================================""\n" ;

    } else {
      amrex::Print() << "Undefine problem type!! Set prob_type in input script." "\n"
                       "prob_type = 1 for 2D problems" "\n"
                       "prob_type = 2 for 3D problems" "\n"
                       "prob_type = 3 for convergence tests." "\n";
      amrex::Abort();
    }

    // Read this from inputs file. Default seed = 1
    int seed = random_seed;

    int nprocs = ParallelDescriptor::NProcs();

    if (prob_type == 1) {
       amrex::InitRandom(seed                             , nprocs, seed                             );  // give all MPI ranks the same seed
    } else { 
      amrex::InitRandom(seed+ParallelDescriptor::MyProc(), nprocs, seed+ParallelDescriptor::MyProc());  // give all MPI ranks a different seed
    }

    int nrand = n_cell[0]*n_cell[2];
    amrex::Gpu::ManagedVector<Real> rngs(nrand, 0.0);

    // generate random numbers on the host
    for (int i=0; i<nrand; ++i) {
        //rngs[i] = amrex::RandomNormal(0.,1.); // zero mean, unit variance
         rngs[i] = amrex::Random(); // uniform [0,1] option
    }

    // loop over boxes
    for (MFIter mfi(rho); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        const Array4<Real> &pOld_p = P_old[0].array(mfi);
        const Array4<Real> &pOld_q = P_old[1].array(mfi);
        const Array4<Real> &pOld_r = P_old[2].array(mfi);
        const Array4<Real>& mat_BigGamma = BigGamma.array(mfi);
        const Array4<Real const>& mask = MaterialMask.array(mfi);
        const Array4<Real const>& tphase = tphaseMask.array(mfi);

        Real* rng = rngs.data();

        // set P
        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];
            if (mask(i,j,k) == 0.0) { //FE mask is 0.0
               if (prob_type == 1) {  //2D : Initialize uniform P in y direction

                 pOld_p(i,j,k) = (-1.0 + 2.0*rng[i + k*n_cell[2]])*Remnant_P[0];
                 pOld_q(i,j,k) = (-1.0 + 2.0*rng[i + k*n_cell[2]])*Remnant_P[1];
                 pOld_r(i,j,k) = (-1.0 + 2.0*rng[i + k*n_cell[2]])*Remnant_P[2];

               } else if (prob_type == 2) { // 3D : Initialize random P

                 pOld_p(i,j,k) = (-1.0 + 2.0*Random(engine))*Remnant_P[0];
                 pOld_q(i,j,k) = (-1.0 + 2.0*Random(engine))*Remnant_P[1];
                 pOld_r(i,j,k) = (-1.0 + 2.0*Random(engine))*Remnant_P[2];

               } else if (prob_type == 3) { // smooth P for convergence tests

                 pOld_p(i,j,k) = Remnant_P[0]*exp(-(x*x/(2.0*5.e-9*5.e-9) + y*y/(2.0*5.e-9*5.e-9) + (z-1.5*DE_hi[2])*(z - 1.5*DE_hi[2])/(2.0*2.0e-9*2.0e-9)));
                 pOld_q(i,j,k) = Remnant_P[1]*exp(-(x*x/(2.0*5.e-9*5.e-9) + y*y/(2.0*5.e-9*5.e-9) + (z-1.5*DE_hi[2])*(z - 1.5*DE_hi[2])/(2.0*2.0e-9*2.0e-9)));
                 pOld_r(i,j,k) = Remnant_P[2]*exp(-(x*x/(2.0*5.e-9*5.e-9) + y*y/(2.0*5.e-9*5.e-9) + (z-1.5*DE_hi[2])*(z - 1.5*DE_hi[2])/(2.0*2.0e-9*2.0e-9)));

               } else {

                 Abort("Invalid prob_type");

               }

	       //set t_phase Pz to zero
	       //if(x <= t_phase_hi[0] && x >= t_phase_lo[0] && y <= t_phase_hi[1] && y >= t_phase_lo[1] && z <= t_phase_hi[2] && z >= t_phase_lo[2]){
	       if(tphase(i,j,k) == 1.0){
                 pOld_r(i,j,k) = 0.0;
	       }

            } else {
               pOld_p(i,j,k) = 0.0;
               pOld_q(i,j,k) = 0.0;
               pOld_r(i,j,k) = 0.0;
               mat_BigGamma(i,j,k) = 0.0; // Note this is overwriting the initialized Gamma, therefore this function must be called after InitializeMaterialProperties
            }

	    if (is_polarization_scalar == 1){
               pOld_p(i,j,k) = 0.0;
               pOld_q(i,j,k) = 0.0;
	    }
        });
        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev

	MultiFab acceptor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);
	MultiFab donor_den(rho.boxArray(), rho.DistributionMap(), 1, 0);

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& acceptor_den_arr = acceptor_den.array(mfi);
        const Array4<Real>& donor_den_arr = donor_den.array(mfi);


        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

             //SC region
             if (mask(i,j,k) >= 2.0) {

                hole_den_arr(i,j,k) = intrinsic_carrier_concentration;
                e_den_arr(i,j,k) = intrinsic_carrier_concentration;
                acceptor_den_arr(i,j,k) = acceptor_doping;
                donor_den_arr(i,j,k) = donor_doping;
             }

             charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));

        });
    }
    for (int i = 0; i < 3; i++){
      // fill periodic ghost cells
      P_old[i].FillBoundary(geom.periodicity());
    }

 }

// create a mask filled with integers to idetify different material types
void InitializeMaterialMask(MultiFab& MaterialMask, 
		                    const Geometry& geom, 
			                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                            const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi)
{
    // loop over boxes
    for (MFIter mfi(MaterialMask); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.growntilebox(MaterialMask.nGrow());
        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        const Array4<Real>& mask = MaterialMask.array(mfi);


        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
             Real x = prob_lo[0] + (i+0.5) * dx[0];
             Real y = prob_lo[1] + (j+0.5) * dx[1];
             Real z = prob_lo[2] + (k+0.5) * dx[2];

             //FE:0, DE:1, Source/Drain:2, Channel:3
             if (x <= FE_hi[0] && x >= FE_lo[0] && y <= FE_hi[1] && y >= FE_lo[1] && z <= FE_hi[2] && z >= FE_lo[2]) {
                 mask(i,j,k) = 0.;
             } else if (x <= DE_hi[0] && x >= DE_lo[0] && y <= DE_hi[1] && y >= DE_lo[1] && z <= DE_hi[2] && z >= DE_lo[2]) {
                 mask(i,j,k) = 1.;
             } else if (x <= DE1_hi[0] && x >= DE1_lo[0] && y <= DE1_hi[1] && y >= DE1_lo[1] && z <= DE1_hi[2] && z >= DE1_lo[2]) {
                 mask(i,j,k) = 1.;
             } else if (x <= SC_hi[0] && x >= SC_lo[0] && y <= SC_hi[1] && y >= SC_lo[1] && z <= SC_hi[2] && z >= SC_lo[2]) {
                 mask(i,j,k) = 2.;
                if (x <= Channel_hi[0] && x >= Channel_lo[0] && y <= Channel_hi[1] && y >= Channel_lo[1] && z <= Channel_hi[2] && z >= Channel_lo[2]){
                    mask(i,j,k) = 3.;
                }
             } else {
	         mask(i,j,k) = 1.; //spacer is DE
	     }
        });
    }
    MaterialMask.FillBoundary(geom.periodicity());
}

// initialization of mask (device geometry) with parser
void InitializeMaterialMask(c_FerroX& rFerroX, const Geometry& geom, MultiFab& MaterialMask)
{ 
    auto& rGprop = rFerroX.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv = MaterialMask.ixType().toIntVect();

    for (MFIter mfi(MaterialMask, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& mask_arr = MaterialMask.array(mfi);
       // const auto& bx = mfi.tilebox();
        const Box& bx = mfi.growntilebox(MaterialMask.nGrow());

	    std::string m_mask_s;
	    std::unique_ptr<amrex::Parser> m_mask_parser;
        std::string m_str_device_geom_function;

	    ParmParse pp_mask("device_geom");


        if (pp_mask.query("device_geom_function(x,y,z)", m_str_device_geom_function) ) {
                m_mask_s = "parse_device_geom_function";
            }

            if (m_mask_s == "parse_device_geom_function") {
                Store_parserString(pp_mask, "device_geom_function(x,y,z)", m_str_device_geom_function);
                m_mask_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_device_geom_function,{"x","y","z"}));
            }

            const auto& macro_parser = m_mask_parser->compile<3>();

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv,macro_parser,mask_arr);
            });

    }
	MaterialMask.FillBoundary(geom.periodicity());
}

// initialization of t-phase mask with parser
void Initialize_tphase_Mask(c_FerroX& rFerroX, const Geometry& geom, MultiFab& tphaseMask)
{ 
    auto& rGprop = rFerroX.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv = tphaseMask.ixType().toIntVect();

    for (MFIter mfi(tphaseMask, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& mask_arr = tphaseMask.array(mfi);
        const auto& bx = mfi.tilebox();

	std::string tphase_mask_s;
	std::unique_ptr<amrex::Parser> tphase_mask_parser;
        std::string m_str_tphase_geom_function;

	ParmParse pp_mask("tphase_geom");


	if (pp_mask.query("tphase_geom_function(x,y,z)", m_str_tphase_geom_function) ) {
            tphase_mask_s = "parse_tphase_geom_function";
        }

        if (tphase_mask_s == "parse_tphase_geom_function") {
            Store_parserString(pp_mask, "tphase_geom_function(x,y,z)", m_str_tphase_geom_function);
            tphase_mask_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_tphase_geom_function,{"x","y","z"}));
        }

        const auto& macro_parser = tphase_mask_parser->compile<3>();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv,macro_parser,mask_arr);
        });

    }
	tphaseMask.FillBoundary(geom.periodicity());
}


// initialization of Euler angles
void Initialize_Euler_angles(c_FerroX& rFerroX, const Geometry& geom, MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta)
{ 
    auto& rGprop = rFerroX.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv_angle_alpha = angle_alpha.ixType().toIntVect();
    const auto iv_angle_beta = angle_beta.ixType().toIntVect();
    const auto iv_angle_theta = angle_theta.ixType().toIntVect();

    for (MFIter mfi(angle_alpha, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& angle_alpha_arr = angle_alpha.array(mfi);
        const auto& angle_beta_arr = angle_beta.array(mfi);
        const auto& angle_theta_arr = angle_theta.array(mfi);
        const auto& bx = mfi.tilebox();

	    std::string angle_alpha_s;
	    std::unique_ptr<amrex::Parser> angle_alpha_parser;
        std::string m_str_angle_alpha_function;

	    std::string angle_beta_s;
	    std::unique_ptr<amrex::Parser> angle_beta_parser;
        std::string m_str_angle_beta_function;

	    std::string angle_theta_s;
	    std::unique_ptr<amrex::Parser> angle_theta_parser;
        std::string m_str_angle_theta_function;

	    ParmParse pp_mask("o_phase_angle");


	    if (pp_mask.query("angle_alpha_function(x,y,z)", m_str_angle_alpha_function) ) {
            angle_alpha_s = "parse_angle_alpha_function";
        }

        if (angle_alpha_s == "parse_angle_alpha_function") {
            Store_parserString(pp_mask, "angle_alpha_function(x,y,z)", m_str_angle_alpha_function);
            angle_alpha_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_angle_alpha_function,{"x","y","z"}));
        }


	    if (pp_mask.query("angle_beta_function(x,y,z)", m_str_angle_beta_function) ) {
            angle_beta_s = "parse_angle_beta_function";
        }

        if (angle_beta_s == "parse_angle_beta_function") {
            Store_parserString(pp_mask, "angle_beta_function(x,y,z)", m_str_angle_beta_function);
            angle_beta_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_angle_beta_function,{"x","y","z"}));
        }

	    if (pp_mask.query("angle_theta_function(x,y,z)", m_str_angle_theta_function) ) {
            angle_theta_s = "parse_angle_theta_function";
        }

        if (angle_theta_s == "parse_angle_theta_function") {
            Store_parserString(pp_mask, "angle_theta_function(x,y,z)", m_str_angle_theta_function);
            angle_theta_parser = std::make_unique<amrex::Parser>(
                                     makeParser(m_str_angle_theta_function,{"x","y","z"}));
        }

        const auto& macro_parser_angle_alpha = angle_alpha_parser->compile<3>();
        const auto& macro_parser_angle_beta = angle_beta_parser->compile<3>();
        const auto& macro_parser_angle_theta = angle_theta_parser->compile<3>();

        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_angle_alpha,macro_parser_angle_alpha,angle_alpha_arr);
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_angle_beta, macro_parser_angle_beta, angle_beta_arr );
            eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_angle_theta,macro_parser_angle_theta,angle_theta_arr);
        });

    }
	angle_alpha.FillBoundary(geom.periodicity());
	angle_beta.FillBoundary(geom.periodicity());
	angle_theta.FillBoundary(geom.periodicity());
}

// initialization of Material Properties
void Initialize_MaterialProperties(c_FerroX& rFerroX, const Geometry& geom, 
                                   MultiFab& BigGamma,
                                   MultiFab& alpha, 
                                   MultiFab& beta, 
                                   MultiFab& gamma, 
                                   MultiFab& epsilonX_fe, 
                                   MultiFab& epsilonZ_fe, 
                                   MultiFab& epsilon_de, 
                                   MultiFab& epsilon_si, 
                                   MultiFab& g11, 
                                   MultiFab& g44, 
                                   MultiFab& g44_p, 
                                   MultiFab& g12, 
                                   MultiFab& alpha_12, 
                                   MultiFab& alpha_112, 
                                   MultiFab& alpha_123)
{ 
    auto& rGprop = rFerroX.get_GeometryProperties();
    Box const& domain = rGprop.geom.Domain();

    const auto dx = rGprop.geom.CellSizeArray();
    const auto& real_box = rGprop.geom.ProbDomain();
    const auto iv_mat_BigGamma = BigGamma.ixType().toIntVect();
    const auto iv_mat_alpha = alpha.ixType().toIntVect();
    const auto iv_mat_beta = beta.ixType().toIntVect();
    const auto iv_mat_gamma = gamma.ixType().toIntVect();
    const auto iv_mat_epsilonX_fe = epsilonX_fe.ixType().toIntVect();
    const auto iv_mat_epsilonZ_fe = epsilonZ_fe.ixType().toIntVect();
    const auto iv_mat_epsilon_de = epsilon_de.ixType().toIntVect();
    const auto iv_mat_epsilon_si = epsilon_si.ixType().toIntVect();
    const auto iv_mat_g11 = g11.ixType().toIntVect();
    const auto iv_mat_g44 = g44.ixType().toIntVect();
    const auto iv_mat_g44_p = g44_p.ixType().toIntVect();
    const auto iv_mat_g12 = g12.ixType().toIntVect();
    const auto iv_mat_alpha_12 = alpha_12.ixType().toIntVect();
    const auto iv_mat_alpha_112 = alpha_112.ixType().toIntVect();
    const auto iv_mat_alpha_123 = alpha_123.ixType().toIntVect();

    for (MFIter mfi(alpha, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const auto& mat_BigGamma_arr = BigGamma.array(mfi);
        const auto& mat_alpha_arr = alpha.array(mfi);
        const auto& mat_beta_arr = beta.array(mfi);
        const auto& mat_gamma_arr = gamma.array(mfi);
        const auto& mat_epsilonX_fe_arr = epsilonX_fe.array(mfi);
        const auto& mat_epsilonZ_fe_arr = epsilonZ_fe.array(mfi);
        const auto& mat_epsilon_de_arr = epsilon_de.array(mfi);
        const auto& mat_epsilon_si_arr = epsilon_si.array(mfi);
        const auto& mat_g11_arr = g11.array(mfi);
        const auto& mat_g44_arr = g44.array(mfi);
        const auto& mat_g44_p_arr = g44_p.array(mfi);
        const auto& mat_g12_arr = g12.array(mfi);
        const auto& mat_alpha_12_arr = alpha_12.array(mfi);
        const auto& mat_alpha_112_arr = alpha_112.array(mfi);
        const auto& mat_alpha_123_arr = alpha_123.array(mfi);
        const auto& bx = mfi.tilebox();

        std::string mat_BigGamma_s;
        std::unique_ptr<amrex::Parser> mat_BigGamma_parser;
        std::string m_str_mat_BigGamma_function;
        
        std::string mat_alpha_s;
        std::unique_ptr<amrex::Parser> mat_alpha_parser;
        std::string m_str_mat_alpha_function;

        std::string mat_beta_s;
        std::unique_ptr<amrex::Parser> mat_beta_parser;
        std::string m_str_mat_beta_function;

        std::string mat_gamma_s;
        std::unique_ptr<amrex::Parser> mat_gamma_parser;
        std::string m_str_mat_gamma_function;

        std::string mat_epsilonX_fe_s;
        std::unique_ptr<amrex::Parser> mat_epsilonX_fe_parser;
        std::string m_str_mat_epsilonX_fe_function;

        std::string mat_epsilonZ_fe_s;
        std::unique_ptr<amrex::Parser> mat_epsilonZ_fe_parser;
        std::string m_str_mat_epsilonZ_fe_function;

        std::string mat_epsilon_de_s;
        std::unique_ptr<amrex::Parser> mat_epsilon_de_parser;
        std::string m_str_mat_epsilon_de_function;

        std::string mat_epsilon_si_s;
        std::unique_ptr<amrex::Parser> mat_epsilon_si_parser;
        std::string m_str_mat_epsilon_si_function;

        std::string mat_g11_s;
        std::unique_ptr<amrex::Parser> mat_g11_parser;
        std::string m_str_mat_g11_function;

        std::string mat_g44_s;
        std::unique_ptr<amrex::Parser> mat_g44_parser;
        std::string m_str_mat_g44_function;

        std::string mat_g44_p_s;
        std::unique_ptr<amrex::Parser> mat_g44_p_parser;
        std::string m_str_mat_g44_p_function;

        std::string mat_g12_s;
        std::unique_ptr<amrex::Parser> mat_g12_parser;
        std::string m_str_mat_g12_function;

        std::string mat_alpha_12_s;
        std::unique_ptr<amrex::Parser> mat_alpha_12_parser;
        std::string m_str_mat_alpha_12_function;

        std::string mat_alpha_112_s;
        std::unique_ptr<amrex::Parser> mat_alpha_112_parser;
        std::string m_str_mat_alpha_112_function;

        std::string mat_alpha_123_s;
        std::unique_ptr<amrex::Parser> mat_alpha_123_parser;
        std::string m_str_mat_alpha_123_function;

        ParmParse pp_mask("material_properties");

        if (pp_mask.query("BigGamma_function(x,y,z)", m_str_mat_BigGamma_function) ) {
                mat_BigGamma_s = "parse_BigGamma_function";
            }
        if (mat_BigGamma_s == "parse_BigGamma_function") {
                Store_parserString(pp_mask, "BigGamma_function(x,y,z)", m_str_mat_BigGamma_function);
                mat_BigGamma_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_BigGamma_function,{"x","y","z"}));
            }

        if (pp_mask.query("landau_alpha_function(x,y,z)", m_str_mat_alpha_function) ) {
                mat_alpha_s = "parse_landau_alpha_function";
            }
        if (mat_alpha_s == "parse_landau_alpha_function") {
                Store_parserString(pp_mask, "landau_alpha_function(x,y,z)", m_str_mat_alpha_function);
                mat_alpha_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_alpha_function,{"x","y","z"}));
            }

        if (pp_mask.query("landau_beta_function(x,y,z)", m_str_mat_beta_function) ) {
                mat_beta_s = "parse_landau_beta_function";
            }
        if (mat_beta_s == "parse_landau_beta_function") {
            Store_parserString(pp_mask, "landau_beta_function(x,y,z)", m_str_mat_beta_function);
            mat_beta_parser = std::make_unique<amrex::Parser>(
                                    makeParser(m_str_mat_beta_function,{"x","y","z"}));
            }

        if (pp_mask.query("landau_gamma_function(x,y,z)", m_str_mat_gamma_function) ) {
                mat_gamma_s = "parse_landau_gamma_function";
            }
        if (mat_gamma_s == "parse_landau_gamma_function") {
                Store_parserString(pp_mask, "landau_gamma_function(x,y,z)", m_str_mat_gamma_function);
                mat_gamma_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_gamma_function,{"x","y","z"}));
            }
        
        if (pp_mask.query("epsilonX_fe_function(x,y,z)", m_str_mat_epsilonX_fe_function) ) {
                mat_epsilonX_fe_s = "parse_epsilonX_fe_function";
            }
        if (mat_epsilonX_fe_s == "parse_epsilonX_fe_function") {
                Store_parserString(pp_mask, "epsilonX_fe_function(x,y,z)", m_str_mat_epsilonX_fe_function);
                mat_epsilonX_fe_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_epsilonX_fe_function,{"x","y","z"}));
            }
        
        if (pp_mask.query("epsilonZ_fe_function(x,y,z)", m_str_mat_epsilonZ_fe_function) ) {
                mat_epsilonZ_fe_s = "parse_epsilonZ_fe_function";
            }
        if (mat_epsilonZ_fe_s == "parse_epsilonZ_fe_function") {
                Store_parserString(pp_mask, "epsilonZ_fe_function(x,y,z)", m_str_mat_epsilonZ_fe_function);
                mat_epsilonZ_fe_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_epsilonZ_fe_function,{"x","y","z"}));
            }
        
        if (pp_mask.query("epsilon_de_function(x,y,z)", m_str_mat_epsilon_de_function) ) {
                mat_epsilon_de_s = "parse_epsilon_de_function";
            }
        if (mat_epsilon_de_s == "parse_epsilon_de_function") {
                Store_parserString(pp_mask, "epsilon_de_function(x,y,z)", m_str_mat_epsilon_de_function);
                mat_epsilon_de_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_epsilon_de_function,{"x","y","z"}));
            }
        
        if (pp_mask.query("epsilon_si_function(x,y,z)", m_str_mat_epsilon_si_function) ) {
                mat_epsilon_si_s = "parse_epsilon_si_function";
            }
        if (mat_epsilon_si_s == "parse_epsilon_si_function") {
                Store_parserString(pp_mask, "epsilon_si_function(x,y,z)", m_str_mat_epsilon_si_function);
                mat_epsilon_si_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_epsilon_si_function,{"x","y","z"}));
            }

        if (pp_mask.query("g11_function(x,y,z)", m_str_mat_g11_function) ) {
                mat_g11_s = "parse_g11_function";
            }
        if (mat_g11_s == "parse_g11_function") {
                Store_parserString(pp_mask, "g11_function(x,y,z)", m_str_mat_g11_function);
                mat_g11_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_g11_function,{"x","y","z"}));
            }

        if (pp_mask.query("g44_function(x,y,z)", m_str_mat_g44_function) ) {
                mat_g44_s = "parse_g44_function";
            }
        if (mat_g44_s == "parse_g44_function") {
                Store_parserString(pp_mask, "g44_function(x,y,z)", m_str_mat_g44_function);
                mat_g44_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_g44_function,{"x","y","z"}));
            }

        if (pp_mask.query("g44_p_function(x,y,z)", m_str_mat_g44_p_function) ) {
                mat_g44_p_s = "parse_g44_p_function";
            }
        if (mat_g44_p_s == "parse_g44_p_function") {
                Store_parserString(pp_mask, "g44_p_function(x,y,z)", m_str_mat_g44_p_function);
                mat_g44_p_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_g44_p_function,{"x","y","z"}));
            }

        if (pp_mask.query("g12_function(x,y,z)", m_str_mat_g12_function) ) {
                mat_g12_s = "parse_g12_function";
            }
        if (mat_g12_s == "parse_g12_function") {
                Store_parserString(pp_mask, "g12_function(x,y,z)", m_str_mat_g12_function);
                mat_g12_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_g12_function,{"x","y","z"}));
            }

        if (pp_mask.query("alpha_12_function(x,y,z)", m_str_mat_alpha_12_function) ) {
                mat_alpha_12_s = "parse_alpha_12_function";
            }
        if (mat_alpha_12_s == "parse_alpha_12_function") {
                Store_parserString(pp_mask, "alpha_12_function(x,y,z)", m_str_mat_alpha_12_function);
                mat_alpha_12_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_alpha_12_function,{"x","y","z"}));
            }

        if (pp_mask.query("alpha_112_function(x,y,z)", m_str_mat_alpha_112_function) ) {
                mat_alpha_112_s = "parse_alpha_112_function";
            }
        if (mat_alpha_112_s == "parse_alpha_112_function") {
                Store_parserString(pp_mask, "alpha_112_function(x,y,z)", m_str_mat_alpha_112_function);
                mat_alpha_112_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_alpha_112_function,{"x","y","z"}));
            }

        if (pp_mask.query("alpha_123_function(x,y,z)", m_str_mat_alpha_123_function) ) {
                mat_alpha_123_s = "parse_alpha_123_function";
            }
        if (mat_alpha_123_s == "parse_alpha_123_function") {
                Store_parserString(pp_mask, "alpha_123_function(x,y,z)", m_str_mat_alpha_123_function);
                mat_alpha_123_parser = std::make_unique<amrex::Parser>(
                                        makeParser(m_str_mat_alpha_123_function,{"x","y","z"}));
            }

        const auto& macro_parser_mat_BigGamma = mat_BigGamma_parser->compile<3>();
        const auto& macro_parser_mat_alpha = mat_alpha_parser->compile<3>();
        const auto& macro_parser_mat_beta = mat_beta_parser->compile<3>();
        const auto& macro_parser_mat_gamma = mat_gamma_parser->compile<3>();
        const auto& macro_parser_mat_epsilonX_fe = mat_epsilonX_fe_parser->compile<3>();
        const auto& macro_parser_mat_epsilonZ_fe = mat_epsilonZ_fe_parser->compile<3>();
        const auto& macro_parser_mat_epsilon_de = mat_epsilon_de_parser->compile<3>();
        const auto& macro_parser_mat_epsilon_si = mat_epsilon_si_parser->compile<3>();
        const auto& macro_parser_mat_g11 = mat_g11_parser->compile<3>();
        const auto& macro_parser_mat_g44 = mat_g44_parser->compile<3>();
        const auto& macro_parser_mat_g44_p = mat_g44_p_parser->compile<3>();
        const auto& macro_parser_mat_g12 = mat_g12_parser->compile<3>();
        const auto& macro_parser_mat_alpha_12 = mat_alpha_12_parser->compile<3>();
        const auto& macro_parser_mat_alpha_112 = mat_alpha_112_parser->compile<3>();
        const auto& macro_parser_mat_alpha_123 = mat_alpha_123_parser->compile<3>();

        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_alpha,macro_parser_mat_alpha,mat_alpha_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_BigGamma,macro_parser_mat_BigGamma,mat_BigGamma_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_beta, macro_parser_mat_beta, mat_beta_arr );
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_gamma,macro_parser_mat_gamma,mat_gamma_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_epsilonX_fe,macro_parser_mat_epsilonX_fe,mat_epsilonX_fe_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_epsilonZ_fe,macro_parser_mat_epsilonZ_fe,mat_epsilonZ_fe_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_epsilon_de,macro_parser_mat_epsilon_de,mat_epsilon_de_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_epsilon_si,macro_parser_mat_epsilon_si,mat_epsilon_si_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_g11,macro_parser_mat_g11,mat_g11_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_g44,macro_parser_mat_g44,mat_g44_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_g44_p,macro_parser_mat_g44_p,mat_g44_p_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_g12,macro_parser_mat_g12,mat_g12_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_alpha_12,macro_parser_mat_alpha_12,mat_alpha_12_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_alpha_112,macro_parser_mat_alpha_112,mat_alpha_112_arr);
                eXstatic_MFab_Util::ConvertParserIntoMultiFab_3vars(i,j,k,dx,real_box,iv_mat_alpha_123,macro_parser_mat_alpha_123,mat_alpha_123_arr);
            });

    }
	BigGamma.FillBoundary(geom.periodicity());
	alpha.FillBoundary(geom.periodicity());
	beta.FillBoundary(geom.periodicity());
	gamma.FillBoundary(geom.periodicity());
	epsilonX_fe.FillBoundary(geom.periodicity());
	epsilonZ_fe.FillBoundary(geom.periodicity());
	epsilon_de.FillBoundary(geom.periodicity());
	epsilon_si.FillBoundary(geom.periodicity());
    g11.FillBoundary(geom.periodicity());
    g44.FillBoundary(geom.periodicity());
    g44_p.FillBoundary(geom.periodicity());
    g12.FillBoundary(geom.periodicity());
    alpha_12.FillBoundary(geom.periodicity());
    alpha_112.FillBoundary(geom.periodicity());
    alpha_123.FillBoundary(geom.periodicity());
}

void SetHardToSwitchNucleation(MultiFab& alpha, MultiFab& NucleationMask, const amrex::GpuArray<int, AMREX_SPACEDIM>& n_cell, amrex::Real hardswitch_ratio, amrex::Real hardswitch_alpha_ratio)
{
    int seed = random_seed;

    int nprocs = ParallelDescriptor::NProcs();

    if (prob_type == 1) {
       amrex::InitRandom(seed                             , nprocs, seed                             );  // give all MPI ranks the same seed
    } else {
      amrex::InitRandom(seed+ParallelDescriptor::MyProc(), nprocs, seed+ParallelDescriptor::MyProc());  // give all MPI ranks a different seed
    }

    int nrand = n_cell[0]*n_cell[2];
    amrex::Gpu::ManagedVector<Real> rngs(nrand, 0.0);

    // generate random numbers on the host
    for (int i=0; i<nrand; ++i) {
        //rngs[i] = amrex::RandomNormal(0.,1.); // zero mean, unit variance
         rngs[i] = amrex::Random(); // uniform [0,1] option
    }
    // printf("Set Nucleation alpha\n");
    for (MFIter mfi(alpha); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        const Array4<Real> &mat_alpha_arr = alpha.array(mfi);
        const Array4<Real>& mask = NucleationMask.array(mfi);

        Real* rng = rngs.data();

        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
               if (mask(i,j,k) == 0.) {
                   if (prob_type == 1) {  //2D
		                if (rng[i + k*n_cell[2]] <= hardswitch_ratio){
                           mat_alpha_arr(i,j,k) = mat_alpha_arr(i,j,k) * hardswitch_alpha_ratio;
		                } else { 
                           mat_alpha_arr(i,j,k) = mat_alpha_arr(i,j,k);
		                }
                   } else if (prob_type == 2) { //3D
                       Real rand = Random(engine);
		                if (rand <= 0.04) {
                            mat_alpha_arr(i,j,k) = mat_alpha_arr(i,j,k) * 10.0;
		                } else {
                          mat_alpha_arr(i,j,k) = mat_alpha_arr(i,j,k);
                        }
		            }
              }
        });
    }

}
