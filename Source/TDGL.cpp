#include "TDGL.H"

// INITIALIZE rho in SC region
void InitializePandRho(int prob_type, 
                   MultiFab&   P_old,
                   MultiFab&   Gamma,
                   MultiFab&   rho,
                   MultiFab&   e_den,
                   MultiFab&   p_den,
                   Real        SC_lo,
                   Real        SC_hi,
                   Real        DE_lo,
                   Real        DE_hi,
		   Real        BigGamma,
                   Real        q,
                   Real        Ec,
                   Real        Ev,
                   Real        kb,
                   Real        T,
                   Real        Nc,
                   Real        Nv,
                   Real        h,
                   Real        m_n,
                   Real        m_p,
                   int         quadrature_order,
                   amrex::GpuArray<amrex::Real, 50> node,
                   amrex::GpuArray<amrex::Real, 50> weight,
                   amrex::GpuArray<amrex::Real, 3> prob_lo,
                   amrex::GpuArray<amrex::Real, 3> prob_hi,
                   const       Geometry& geom)
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
        
	const Array4<Real>& pOld = P_old.array(mfi);
        const Array4<Real>& Gam = Gamma.array(mfi);

	Real pi = 3.141592653589793238;


        // set P
        amrex::ParallelForRNG(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k, amrex::RandomEngine const& engine) noexcept
        {
            Real x = prob_lo[0] + (i+0.5) * dx[0];
            Real y = prob_lo[1] + (j+0.5) * dx[1];
            Real z = prob_lo[2] + (k+0.5) * dx[2];
            if (z <= DE_hi) {
               pOld(i,j,k) = 0.0;
               Gam(i,j,k) = 0.0;
            } else {
               if (prob_type == 1) {  //2D : Initialize uniform P in y direction

                  double tmp = (i%3 + k%4)/5.;
                  pOld(i,j,k) = (-1.0 + 2.0*tmp)*0.002;

               } else if (prob_type == 2) { // 3D : Initialize random P

                 pOld(i,j,k) = (-1.0 + 2.0*Random(engine))*0.002;

               } else if (prob_type == 3) { // smooth P for convergence tests

	         pOld(i,j,k) = 0.002*exp(-(x*x/(2.0*5.e-9*5.e-9) + y*y/(2.0*5.e-9*5.e-9) + (z-1.5*DE_hi)*(z - 1.5*DE_hi)/(2.0*2.0e-9*2.0e-9)));

               } else {

		 Abort("Invalid prob_type");

	       }

               Gam(i,j,k) = BigGamma;
            }
        });

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev 

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
             Real z = prob_lo[2] + (k+0.5) * dx[2];

             if(z <= SC_hi){ //SC region

                //Real qPhi = 0.5*(Ec + Ev); //eV
//                Real qPhi = 0.5*(Ec + Ev); //eV
//                hole_den_arr(i,j,k) = Nv*exp(-(qPhi - Ev)*1.602e-19/(kb*T));
//                e_den_arr(i,j,k) = Nc*exp(-(Ec - qPhi)*1.602e-19/(kb*T));
//                charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));

                //Fermi-Dirac
//                Real n_coeff = 8.0*3.14*m_n*kb*T*sqrt(2.0*m_n*kb*T)/h/h/h; //SI
//                Real p_coeff = 8.0*3.14*m_p*kb*T*sqrt(2.0*m_p*kb*T)/h/h/h; //SI
                Real qPhi = Ev; //0.5*(Ec + Ev); //eV
                Real n_coef1 = 2.0*m_n*kb*T/h/h;
                Real p_coef1 = 2.0*m_p*kb*T/h/h;
                Real n_coeff = 4.0*3.14*std::pow(n_coef1,1.5);
                Real p_coeff = 4.0*3.14*std::pow(p_coef1,1.5);

//                for (int ii = 0; ii < quadrature_order; ii++)
//                {
//
//                    e_den_arr(i,j,k) += n_coeff*weight[ii]*exp(node[ii])/(1.0 + exp(node[ii] - 1.602e-19*(qPhi - Ec)/(kb*T)));
//                    hole_den_arr(i,j,k) += p_coeff*weight[ii]*exp(node[ii])/(1.0 + exp(node[ii] - 1.602e-19*(Ev - qPhi)/(kb*T)));
//                    //hole_den_arr(i,j,k) += p_coeff*weight[ii]*(exp(1.602e-19*(Ev - qPhi)/(kb*T))/(1.0 + exp(-node[ii] + 1.602e-19*(Ev - qPhi)/(kb*T))));
//               
//                }
                    //hole_den_arr(i,j,k) = Nv*exp(-(qPhi - Ev)*1.602e-19/(kb*T));
                  //amrex::Print() << "ne = " << e_den_arr(i,j,k) << ", np = " << hole_den_arr(i,j,k) << "\n" ;
                    //Approximate FD integral
                    Real eta_n = q*(qPhi - Ec)/(kb*T);
                    Real nu_n = std::pow(eta_n, 4.0) + 50.0 + 33.6 * eta_n * (1 - 0.68 * exp(-0.17 * std::pow((eta_n + 1), 2)));
                    Real xi_n = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_n, 3/8));
                    Real FD_half_n = std::pow(exp(-eta_n) + xi_n, -1.0);

                    e_den_arr(i,j,k) = 2.0/sqrt(3.14)*Nc*FD_half_n;

                    Real eta_p = q*(Ev - qPhi)/(kb*T);
                    Real nu_p = std::pow(eta_p, 4.0) + 50.0 + 33.6 * eta_p * (1 - 0.68 * exp(-0.17 * std::pow((eta_p + 1), 2)));
                    Real xi_p = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_p, 3/8));
                    Real FD_half_p = std::pow(exp(-eta_p) + xi_p, -1.0);

                    hole_den_arr(i,j,k) = 2.0/sqrt(3.14)*Nv*FD_half_p;
                    charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));

	     } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
    // fill periodic ghost cells
    P_old.FillBoundary(geom.periodicity());

 }


// Compute rho in SC region for given phi
void ComputeRho(MultiFab&      PoissonPhi,
                MultiFab&      rho,
                MultiFab&      e_den,
                MultiFab&      p_den,
                Real           Sc_lo,
                Real           SC_hi,
                Real           q, 
                Real           Ec,
                Real           Ev,
                Real           kb,
                Real           T,
                Real           Nc,
                Real           Nv,
                Real        h,
                Real        m_n,
                Real        m_p,
                int         quadrature_order,
                amrex::GpuArray<amrex::Real, 50> node,
                amrex::GpuArray<amrex::Real, 50> weight,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
                const          Geometry& geom)
{
    // loop over boxes
    for (MFIter mfi(PoissonPhi); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

       // extract dx from the geometry object
       GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

        // Calculate charge density from Phi, Nc, Nv, Ec, and Ev 

        const Array4<Real>& hole_den_arr = p_den.array(mfi);
        const Array4<Real>& e_den_arr = e_den.array(mfi);
        const Array4<Real>& charge_den_arr = rho.array(mfi);
        const Array4<Real>& phi = PoissonPhi.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
             Real z = prob_lo[2] + (k+0.5) * dx[2];

             if(z <= SC_hi){ //SC region

//                hole_den_arr(i,j,k) = Nv*exp(-(q*phi(i,j,k) - Ev*1.602e-19)/(kb*T));
//                e_den_arr(i,j,k) = Nc*exp(-(Ec*1.602e-19 - q*phi(i,j,k))/(kb*T));
//                charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));

                //Fermi-Dirac
//                Real n_coeff = 8.0*3.14*m_n*kb*T*sqrt(2.0*m_n*kb*T)/h/h/h; //SI
//                Real p_coeff = 8.0*3.14*m_p*kb*T*sqrt(2.0*m_p*kb*T)/h/h/h; //SI
//                for (int ii = 0; ii < quadrature_order; ii++)
//                {
//
//                    e_den_arr(i,j,k) += n_coeff*weight[ii]*exp(node[ii])/(1.0 + exp(node[ii] - (q*phi(i,j,k) - Ec*1.602e-19)/(kb*T)));
//                   // hole_den_arr(i,j,k) += p_coeff*weight[ii]*exp(node[ii])*(1.0 - 1.0/(1.0 + exp(node[ii] + 1.602e-19*(q*phi(i,j,k) - Ev)/(kb*T))));
//                   hole_den_arr(i,j,k) += p_coeff*weight[ii]*(exp((1.602e-19*Ev - q*phi(i,j,k))/(kb*T))/(1.0 + exp(-node[ii] + (1.602e-19*Ev - q*phi(i,j,k))/(kb*T))));
                     Real n_coef1 = 2.0*m_n*kb*T/h/h;
                     Real p_coef1 = 2.0*m_p*kb*T/h/h;
                     Real n_coeff = 4.0*3.14*std::pow(n_coef1,1.5);
                     Real p_coeff = 4.0*3.14*std::pow(p_coef1,1.5);

//                     for (int ii = 0; ii < quadrature_order; ii++)
//                     {
//
//                         e_den_arr(i,j,k) += n_coeff*weight[ii]*exp(node[ii])/(1.0 + exp(node[ii] - q*(phi(i,j,k) - Ec)/(kb*T)));
//                         hole_den_arr(i,j,k) += p_coeff*weight[ii]*exp(node[ii])/(1.0 + exp(node[ii] - q*(Ev - phi(i,j,k))/(kb*T)));
//                     }
                    //hole_den_arr(i,j,k) = Nv*exp(-(q*phi(i,j,k) - Ev*1.602e-19)/(kb*T));
                    //amrex::Print() << "ne = " << e_den_arr(i,j,k) << ", np = " << hole_den_arr(i,j,k) << "\n" ;
                    //Approximate FD integral
                    Real eta_n = q*(phi(i,j,k) - Ec)/(kb*T);
                    Real nu_n = std::pow(eta_n, 4.0) + 50.0 + 33.6 * eta_n * (1 - 0.68 * exp(-0.17 * std::pow((eta_n + 1), 2)));
                    Real xi_n = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_n, 3/8));
                    Real FD_half_n = std::pow(exp(-eta_n) + xi_n, -1.0);

                    e_den_arr(i,j,k) = 2.0/sqrt(3.14)*Nc*FD_half_n;

                    Real eta_p = q*(Ev - phi(i,j,k))/(kb*T);
                    Real nu_p = std::pow(eta_p, 4.0) + 50.0 + 33.6 * eta_p * (1 - 0.68 * exp(-0.17 * std::pow((eta_p + 1), 2)));
                    Real xi_p = 3.0 * sqrt(3.14)/(4.0 * std::pow(nu_p, 3/8));
                    Real FD_half_p = std::pow(exp(-eta_p) + xi_p, -1.0);

                    hole_den_arr(i,j,k) = 2.0/sqrt(3.14)*Nv*FD_half_p;

                    charge_den_arr(i,j,k) = q*(hole_den_arr(i,j,k) - e_den_arr(i,j,k));
             } else {

                charge_den_arr(i,j,k) = 0.0;

             }
        });
    }
 }

void ComputePoissonRHS(MultiFab&               PoissonRHS,
                MultiFab&                      P_old,
                MultiFab&                      rho,
                Real                           FE_lo,
                Real                           FE_hi,
                Real                           DE_lo,
                Real                           DE_hi,
                Real                           SC_lo,
                Real                           SC_hi,
                int                            P_BC_flag_lo,
                int                            P_BC_flag_hi,
                Real                           lambda,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
                const Geometry&                 geom)
{
    for ( MFIter mfi(PoissonRHS); mfi.isValid(); ++mfi )
        {
            const Box& bx = mfi.validbox();
            // extract dx from the geometry object
            GpuArray<Real,AMREX_SPACEDIM> dx = geom.CellSizeArray();

            const Array4<Real>& pOld = P_old.array(mfi);
            const Array4<Real>& RHS = PoissonRHS.array(mfi);
            const Array4<Real>& charge_den_arr = rho.array(mfi);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                 Real z    = prob_lo[2] + (k+0.5) * dx[2];
                 Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                 Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

                 if(z <= SC_hi){ //SC region

                   RHS(i,j,k) = charge_den_arr(i,j,k);

                 } else if(z < DE_hi){ //DE region

                   RHS(i,j,k) = 0.;

                 } else if (DE_hi > z_lo && DE_hi <= z) { //FE side of FE-DE interface

                   if(P_BC_flag_lo == 0){
                     Real P_int = 0.0;
                     RHS(i,j,k) = -(-4.*P_int + 3.*pOld(i,j,k) + pOld(i,j,k+1))/(3.*dx[2]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k+1)
                   } else if (P_BC_flag_lo == 1){
                     Real P_int = pOld(i,j,k)/(1 + dx[2]/2/lambda);
                     Real dPdz = P_int/lambda;
                     RHS(i,j,k) = -(dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/(2.*dx[2]);
                   } else if (P_BC_flag_lo == 2){
                     Real dPdz = 0.;
                     RHS(i,j,k) = -(dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/(2.*dx[2]);
                   }

                 } else if (z_hi > prob_hi[2]){ //Top metal
                    
	             if(P_BC_flag_hi == 0){
                     Real P_int = 0.0;
                     RHS(i,j,k) = -(4.*P_int - 3.*pOld(i,j,k) - pOld(i,j,k-1))/(3.*dx[2]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k-1)
                     } else if (P_BC_flag_hi == 1){
                     Real P_int = pOld(i,j,k)/(1 - dx[2]/2/lambda);
                     Real dPdz = P_int/lambda;
                     RHS(i,j,k) = -(dx[2]*dPdz + pOld(i,j,k) - pOld(i,j,k-1))/(2.*dx[2]);
                     } else if (P_BC_flag_hi == 2){
                     Real dPdz = 0.;
                     RHS(i,j,k) = -(dx[2]*dPdz + pOld(i,j,k) - pOld(i,j,k-1))/(2.*dx[2]);
                   }

                 } else{ //inside FE

                   RHS(i,j,k) = -(pOld(i,j,k+1) - pOld(i,j,k-1))/(2.*dx[2]);

                 }

            });
        }
   
}

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
                    Real P_int = pOld(i,j,k)/(1 + dx[2]/2/lambda);
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
                    Real P_int = pOld(i,j,k)/(1 - dx[2]/2/lambda);
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


void ComputeEfromPhi(MultiFab&                 PoissonPhi,
                MultiFab&                      Ex,
                MultiFab&                      Ey,
                MultiFab&                      Ez,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
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
                     Ex_arr(i,j,k) = -(phi(i+1,j,k) - phi(i-1,j,k))/(2.*dx[0]);
                     Ey_arr(i,j,k) = -(phi(i,j+1,k) - phi(i,j-1,k))/(2.*dx[1]);

                     Real z    = prob_lo[2] + (k+0.5) * dx[2];
                     Real z_hi = prob_lo[2] + (k+1.5) * dx[2];
                     Real z_lo = prob_lo[2] + (k-0.5) * dx[2];

                     if(z_lo < prob_lo[2]){ //Bottom Boundary
                       Ez_arr(i,j,k) = -(phi(i,j,k+1) - phi(i,j,k))/(dx[2]);
                     } else if (z_hi > prob_hi[2]){ //Top Boundary
                       Ez_arr(i,j,k) = -(phi(i,j,k) - phi(i,j,k-1))/(dx[2]);
                     } else{ //inside
                       Ez_arr(i,j,k) = -(phi(i,j,k+1) - phi(i,j,k-1))/(2.*dx[2]);
                     }
             });
        }

}


void InitializePermittivity(std::array< MultiFab, AMREX_SPACEDIM >& beta_face,
                Real                            FE_lo,
                Real                            FE_hi,
                Real                            DE_lo,
                Real                            DE_hi,
                Real                            SC_lo,
                Real                            SC_hi,
		Real 				epsilon_0,
		Real 				epsilonX_fe,
		Real 				epsilonZ_fe,
		Real 				epsilon_de,
		Real 				epsilon_si,
                amrex::GpuArray<amrex::Real, 3> prob_lo,
                amrex::GpuArray<amrex::Real, 3> prob_hi,
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
          if(z < SC_hi-small) {
             beta_f0(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  } else if(z >= SC_hi-small && z < SC_hi+small){
             beta_f0(i,j,k) = 0.5*(epsilon_de + epsilon_si) * epsilon_0; //SC-DE interface
          } else if(z < DE_hi-small) {
             beta_f0(i,j,k) = epsilon_de * epsilon_0; //DE layer
	  } else if(z >= DE_hi-small && z < DE_hi+small){
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
          if(z < SC_hi-small) {
             beta_f1(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  } else if(z >= SC_hi-small && z < SC_hi+small){
             beta_f1(i,j,k) = 0.5*(epsilon_de + epsilon_si) * epsilon_0; //SC-DE interface
          } else if(z < DE_hi-small) {
             beta_f1(i,j,k) = epsilon_de * epsilon_0; //DE layer
	  } else if(z >= DE_hi-small && z < DE_hi+small){
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
          if(z < SC_hi-small) {
             beta_f2(i,j,k) = epsilon_si * epsilon_0; //SC layer
	  } else if(z >= SC_hi-small && z < SC_hi+small && SC_hi > prob_lo[2]+small){
             beta_f2(i,j,k) = 0.5*(epsilon_de + epsilon_si) * epsilon_0; //SC-DE interface
          } else if(z < DE_hi-small) {
             beta_f2(i,j,k) = epsilon_de * epsilon_0; //DE layer
	  } else if(z >= DE_hi-small && z < DE_hi+small){
             beta_f2(i,j,k) = 0.5*(epsilon_de + epsilonZ_fe) * epsilon_0; //DE-FE interface
             //beta_f2(i,j,k) = epsilon_de * epsilon_0; //DE-FE interface
          } else {
             beta_f2(i,j,k) = epsilonZ_fe * epsilon_0; //FE layer
          }
        });
    }

}

void SetPhiBC_z(MultiFab& PoissonPhi,
                amrex::GpuArray<int, 3> n_cell,
                Real                    Phi_Bc_lo,
                Real                    Phi_Bc_hi)
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
