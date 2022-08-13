#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>

using namespace amrex;

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
                const Geometry& geom);

// /**
//   * Perform double derivative (d^2)F/dz^2 */
//  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
//  static amrex::Real DoubleDPDz (
//     amrex::Array4<amrex::Real> const& F,
//     amrex::Real const z, amrex::Real const z_hi, amrex::Real const z_lo, 
//     int const P_BC_flag_lo, int const P_BC_flag_hi,
//     amrex::Real const FE_lo, amrex::Real const FE_hi,
//     amrex::Real lambda,
//     int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {

//     amrex::Real const inv_dz = 1./dx[2];

//     if (FE_lo > z_lo && FE_lo <= z) { //FE lower boundary
//         if(P_BC_flag_lo == 0){
//             Real F_lo = 0.0;
//             return inv_dz * inv_dz * 4./3. * (2.*F_lo - 3.*F(i,j,k) + F(i,j,k+1));//2nd Order
//         } else if (P_BC_flag_lo == 1){
//             Real F_lo = (9. * F(i,j,k) - F(i,j,k+1)) / (3. * dx[2] / lambda + 8.);
//             return inv_dz * (inv_dz * (F(i,j,k+1) - F(i,j,k)) - F_lo/lambda); // dPdz = P_lo/lambda;
//         } else if (P_BC_flag_lo == 2){
//             return inv_dz * inv_dz * (F(i,j,k+1) - F(i,j,k)); //dPdz = 0.
//         } else {
//             amrex::Abort("Wrong flag of the lower polarization boundary condition!!");
//             return 0.;
//         }
//     } else if (z_hi > FE_hi && z <= FE_hi ){ // FE higher boundary
//         if(P_BC_flag_hi == 0){
//             Real F_hi = 0.0;
//             return inv_dz * inv_dz * 4./3. * (2.*F_hi - 3.*F(i,j,k) + F(i,j,k-1));//2nd Order
//         } else if (P_BC_flag_hi == 1){
//             Real F_hi = (9. * F(i,j,k) - F(i,j,k-1)) / ( - 3. * dx[2] / lambda + 8.);
//             return inv_dz * (inv_dz * (F(i,j,k-1) - F(i,j,k)) + F_hi/lambda); // dPdz = P_hi/lambda;
//         } else if (P_BC_flag_hi == 2){
//             return inv_dz * inv_dz * (F(i,j,k-1) - F(i,j,k)); //dPdz = 0.
//         } else {
//             amrex::Abort("Wrong flag of the higher polarization boundary condition!!");
//             return 0.;
//         }
        
//     } else if (z_hi <= FE_hi && z_lo >= FE_lo) { // inside FE
//         return inv_dz * inv_dz * (F(i,j,k+1) - 2.*F(i,j,k) + F(i,j,k-1));
//     } else {
//         return 0.;
//     }
//  }

//  /**
//   * Perform first derivative dF/dz */
//  AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
//  static amrex::Real DPhiDz (
//     amrex::Array4<amrex::Real> const& F,
//     amrex::Real const z, amrex::Real const z_hi, amrex::Real const z_lo, 
//     amrex::GpuArray<amrex::Real, 3> prob_lo, amrex::GpuArray<amrex::Real, 3> prob_hi,
//     amrex::Real const Phi_Bc_hi, amrex::Real const Phi_Bc_lo,
//     int const i, int const j, int const k,  amrex::GpuArray<amrex::Real, 3> dx) {

//     amrex::Real const inv_dz = 1./dx[2];

//     if (z_lo < prob_lo[2]){ // bottom metal
//         return (1./3.) * inv_dz * (-4.*Phi_Bc_lo + 3.*F(i,j,k) + F(i,j,k+1));

//     } else if (z_hi > prob_hi[2]){ // top metal
//         return (1./3.) * inv_dz * (4.*Phi_Bc_hi - 3.*F(i,j,k) - F(i,j,k-1));

//     } else { // inside stack
//         return (1./2.) * inv_dz * (F(i,j,k+1) - F(i,j,k-1)) ;
//     }
//  }

/**
  * Perform double derivative (d^2)F/dz^2 */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DoubleDPDz (
    amrex::Array4<amrex::Real> const& pOld,
    amrex::Real const z, amrex::Real const z_hi, amrex::Real const z_lo, 
    int const P_BC_flag_lo, int const P_BC_flag_hi,
    amrex::Real const FE_lo, amrex::Real const FE_hi,  amrex::Real const DE_hi,
     amrex::GpuArray<amrex::Real, 3> prob_lo, amrex::GpuArray<amrex::Real, 3> prob_hi,
    amrex::Real lambda,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
        
        amrex::Real const inv_dz = 1./dx[2];
        
        if(z_lo < prob_lo[2]){ //Bottom metal

                  return 0.0;

                } else if(z < DE_hi){ //Below FE-DE interface

                 return 0.0;

                } else if (DE_hi > z_lo && DE_hi <= z) { //FE side of FE-DE interface
                
                  if(P_BC_flag_lo == 0){
                    Real P_int = 0.0;
                    return 4.*(2.*P_int - 3.*pOld(i,j,k) + pOld(i,j,k+1))/3./dx[2]/dx[2];//2nd Order
                  } else if (P_BC_flag_lo == 1){
                    
                    Real P_int = ( 9.*pOld(i,j,k) - pOld(i,j,k+1) ) / (8. + 3.*dx[2] / lambda);
                    Real dPdz = P_int/lambda;
                    return (-dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/dx[2]/dx[2];//2nd Order
                  } else if (P_BC_flag_lo == 2){
                    Real dPdz = 0.;
                    return (-dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k+1))/dx[2]/dx[2];//2nd Order
                  } else{
                    return 0.0;
                  }

                } else if (z_hi > prob_hi[2]){ //Top metal

		  if(P_BC_flag_hi == 0){
                    Real P_int = 0.0;
                    return 4.*(2.*P_int - 3.*pOld(i,j,k) + pOld(i,j,k-1))/3./dx[2]/dx[2];//2nd Order
                  } else if (P_BC_flag_hi == 1){
                    // std::cout << "flag" << P_BC_flag_hi;
		    Real P_int = ( 9.*pOld(i,j,k) - pOld(i,j,k-1) ) / (8. - 3.*dx[2] / lambda);
                    Real dPdz = P_int/lambda;
                    return (dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k-1))/dx[2]/dx[2];//2nd Order
                  } else if (P_BC_flag_hi == 2){
                    Real dPdz = 0.;
                    return (dx[2]*dPdz - pOld(i,j,k) + pOld(i,j,k-1))/dx[2]/dx[2];//2nd Order
                  } else {
                    // std::cout << "see if reach here at all";
                    return 0.0;
                  }

                } else{ //inside FE

                  return (pOld(i,j,k+1) - 2.*pOld(i,j,k) + pOld(i,j,k-1)) / (dx[2]*dx[2]);

                }
    }

     /**
  * Perform first derivative dF/dz */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DPhiDz (
    amrex::Array4<amrex::Real> const& phi,
    amrex::Real const z, amrex::Real const z_hi, amrex::Real const z_lo, 
    amrex::GpuArray<amrex::Real, 3> prob_lo, amrex::GpuArray<amrex::Real, 3> prob_hi,
    amrex::Real const Phi_Bc_hi, amrex::Real const Phi_Bc_lo, amrex::Real const DE_hi,
    int const i, int const j, int const k,  amrex::GpuArray<amrex::Real, 3> dx) {
        
        amrex::Real const inv_dz = 1./dx[2];
        
        if(z_lo < prob_lo[2]){ //Bottom metal

                  return (-4.*Phi_Bc_lo + 3.*phi(i,j,k) + phi(i,j,k+1))/(3.*dx[2]);
                  //phi_term = (phi(i,j,k+1) - phi(i,j,k)) / (dx[2]);

                } else if(z < DE_hi){ //Below FE-DE interface

                  return (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2]);

                } else if (DE_hi > z_lo && DE_hi <= z) { //FE side of FE-DE interface

                  return (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2]);

                } else if (z_hi > prob_hi[2]){ //Top metal

		          return (4.*Phi_Bc_hi - 3.*phi(i,j,k) - phi(i,j,k-1))/(3.*dx[2]);

                } else{ //inside FE

                  return (phi(i,j,k+1) - phi(i,j,k-1)) / (2.*dx[2]);

                }


 }