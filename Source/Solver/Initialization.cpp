#include "Initialization.H"
#include "Utils/eXstaticUtils/eXstaticUtil.H"
#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"

// INITIALIZE rho in SC region
void InitializePandRho(Array<MultiFab, AMREX_SPACEDIM> &P_old,
                   MultiFab&   Gamma,
                   MultiFab&   rho,
                   MultiFab&   e_den,
                   MultiFab&   p_den,
		   const MultiFab& MaterialMask,
		   const MultiFab& tphaseMask,
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

    // loop over boxes
    for (MFIter mfi(rho); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

        // extract dx from the geometry object
        GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        const Array4<Real> &pOld_x = P_old[0].array(mfi);
        const Array4<Real> &pOld_y = P_old[1].array(mfi);
        const Array4<Real> &pOld_z = P_old[2].array(mfi);
        const Array4<Real>& Gam = Gamma.array(mfi);
        const Array4<Real const>& mask = MaterialMask.array(mfi);
        const Array4<Real const>& tphase = tphaseMask.array(mfi);

        Real pi = 3.141592653589793238;

        // set P
        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];
            if (mask(i,j,k) == 0.0) { //FE mask is 0.0
               if (prob_type == 1) {  //2D : Initialize uniform P in y direction

                  double tmp = (i%3 + k%4)/5.;
                  pOld_z(i,j,k) = (-1.0 + 2.0*tmp)*0.002;

               } else if (prob_type == 2) { // 3D : Initialize random P

                 pOld_z(i,j,k) = (-1.0 + 2.0*Random(engine))*0.002;

               } else if (prob_type == 3) { // smooth P for convergence tests

                 pOld_z(i,j,k) = 0.002*exp(-(x*x/(2.0*5.e-9*5.e-9) + y*y/(2.0*5.e-9*5.e-9) + (z-1.5*DE_hi[2])*(z - 1.5*DE_hi[2])/(2.0*2.0e-9*2.0e-9)));

               } else {

                 Abort("Invalid prob_type");

               }

               Gam(i,j,k) = BigGamma;

	       //set t_phase Pz to zero
	       //if(x <= t_phase_hi[0] && x >= t_phase_lo[0] && y <= t_phase_hi[1] && y >= t_phase_lo[1] && z <= t_phase_hi[2] && z >= t_phase_lo[2]){
	       if(tphase(i,j,k) == 1.0){
                 pOld_z(i,j,k) = 0.0;
	       }

            } else {
               pOld_z(i,j,k) = 0.0;
               Gam(i,j,k) = 0.0;
            }
            pOld_x(i,j,k) = 0.0;
            pOld_y(i,j,k) = 0.0;
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

                  Real Phi = 0.5*(Ec + Ev); //eV
//                hole_den_arr(i,j,k) = Nv*exp(-(Phi - Ev)*1.602e-19/(kb*T));
//                e_den_arr(i,j,k) = Nc*exp(-(Ec - Phi)*1.602e-19/(kb*T));
//                charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));

                  //Approximate FD integral
                  Real eta_n = q*(Phi - Ec)/(kb*T);
                  Real nu_n = std::pow(eta_n, 4.0) + 50.0 + 33.6 * eta_n * (1 - 0.68 * exp(-0.17 * std::pow((eta_n + 1), 2.0)));
                  Real xi_n = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_n, 3/8));
                  Real FD_half_n = std::pow(exp(-eta_n) + xi_n, -1.0);

                  e_den_arr(i,j,k) = 2.0/sqrt(3.14)*Nc*FD_half_n;

                  Real eta_p = q*(Ev - Phi)/(kb*T);
                  Real nu_p = std::pow(eta_p, 4.0) + 50.0 + 33.6 * eta_p * (1 - 0.68 * exp(-0.17 * std::pow((eta_p + 1), 2.0)));
                  Real xi_p = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_p, 3/8));
                  Real FD_half_p = std::pow(exp(-eta_p) + xi_p, -1.0);

                  hole_den_arr(i,j,k) = 2.0/sqrt(3.14)*Nv*FD_half_p;

		  //If in channel, set acceptor doping, else (Source/Drain) set donor doping
                  if (mask(i,j,k) == 3.0) {
		     acceptor_den_arr(i,j,k) = acceptor_doping; 
	             donor_den_arr(i,j,k) = 0.0;
		  } else { // Source / Drain
		     acceptor_den_arr(i,j,k) = 0.0; 
	             donor_den_arr(i,j,k) = donor_doping;
		  }
                  charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k) - acceptor_den_arr(i,j,k) + donor_den_arr(i,j,k));

             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
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
        const Box& bx = mfi.validbox();

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
        const auto& bx = mfi.tilebox();

	std::string m_mask_s;
	std::unique_ptr<amrex::Parser> m_mask_parser;
        std::string m_str_device_geom_function;

	ParmParse pp_mask("device_geom");

	bool mask_specified = false;

	if (pp_mask.query("device_geom_function(x,y,z)", m_str_device_geom_function) ) {
            m_mask_s = "parse_device_geom_function";
            mask_specified = true;
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

	bool mask_specified = false;

	if (pp_mask.query("tphase_geom_function(x,y,z)", m_str_tphase_geom_function) ) {
            tphase_mask_s = "parse_tphase_geom_function";
            mask_specified = true;
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

