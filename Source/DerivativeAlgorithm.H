#include "FerroX.H"

using namespace FerroX;

/**
  * Perform first derivative dphi/dz */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DphiDz (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const z, amrex::Real const z_hi, amrex::Real const z_lo, 
    int const i, int const j, int const k,  amrex::GpuArray<amrex::Real, 3> dx) {

    if (z_lo < prob_lo[2]){ // bottom metal
        // return (1./3.) * inv_dz * (-4.*Phi_Bc_lo + 3.*F(i,j,k) + F(i,j,k+1));
        return (-4.*Phi_Bc_lo + 3.*F(i,j,k) + F(i,j,k+1)) / (3. * dx[2]);
        
    } else if (z_hi > prob_hi[2]){ // top metal
        return (4.*Phi_Bc_hi - 3.*F(i,j,k) - F(i,j,k-1)) / (3. * dx[2]);

    } else { // inside stack
        return (F(i,j,k+1) - F(i,j,k-1)) / (2. * dx[2]);
    }
 }

/**
  * Perform first derivative dF/dx */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DFDx (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
    return (F(i+1,j,k) - F(i-1,j,k))/(2.*dx[0]);
 }

 /**
  * Perform first derivative dF/dy */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DFDy (
    amrex::Array4<amrex::Real> const& F,
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
    return (F(i,j+1,k) - F(i,j-1,k))/(2.*dx[1]);
 }

/**
  * Perform first derivative dP/dz */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DPDz (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const z, amrex::Real const z_hi, amrex::Real const z_lo, 
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx
) {

    if (FE_lo[2] > z_lo && FE_lo[2] <= z) { //FE lower boundary
      
        if(P_BC_flag_lo == 0){
            Real F_lo = 0.0;
            return (-4.*F_lo + 3.*F(i,j,k) + F(i,j,k+1))/(3.*dx[2]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k+1)

        } else if (P_BC_flag_lo == 1){

            Real F_lo = F(i,j,k)/(1 + dx[2]/2/lambda);
            return (dx[2]*F_lo/lambda - F(i,j,k) + F(i,j,k+1))/(2.*dx[2]); // dP/dz = P_lo/lambda;

            // Real F_lo = (9. * F(i,j,k) - F(i,j,k+1)) / (3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return  -(dx[2]*F_lo/lambda - F(i,j,k) + F(i,j,k+1))/(2.*dx[2]);// dP/dz = P_lo/lambda;

        } else if (P_BC_flag_lo == 2){
            return ( - F(i,j,k) + F(i,j,k+1))/(2.*dx[2]); //dPdz = 0.

        } else {
            amrex::Abort("Wrong flag of the lower polarization boundary condition!!");
            return 0.0;
        }     

    } else if (z_hi > FE_hi[2] && z <= FE_hi[2] ){ // FE higher boundary

        if(P_BC_flag_hi == 0){
            Real F_hi = 0.0;
            return (4.*F_hi - 3.*F(i,j,k) - F(i,j,k-1))/(3.*dx[2]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k-1)

        } else if (P_BC_flag_hi == 1){
            
            Real F_hi = F(i,j,k)/(1 - dx[2]/2/lambda);
            return (dx[2]*F_hi/lambda + F(i,j,k) - F(i,j,k-1))/(2.*dx[2]);//dPdz = P_hi/lambda;

            // Real F_hi = (9. * F(i,j,k) - F(i,j,k-1)) / ( - 3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return  -(dx[2]*F_hi/lambda + F(i,j,k) - F(i,j,k-1))/(2.*dx[2]);//dPdz = P_hi/lambda;

        } else if (P_BC_flag_hi == 2){
            return (F(i,j,k) - F(i,j,k-1))/(2.*dx[2]); //dPdz = 0.

        } else {
            amrex::Abort("Wrong flag of the higher polarization boundary condition!!");
            return 0.0;
        }
                  
    } else if (z_hi <= FE_hi[2] && z_lo >= FE_lo[2]) { // inside FE
        return (F(i,j,k+1) - F(i,j,k-1))/(2.*dx[2]);

    } else {
        return 0.0;
    }
 }

/**
  * Perform double derivative (d^2)P/dx^2 */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DoubleDPDx (
    amrex::Array4<amrex::Real> const& F, 
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
        return (F(i+1,j,k) - 2.*F(i,j,k) + F(i-1,j,k)) / (dx[0]*dx[0]);  
               
}

/**
  * Perform double derivative (d^2)P/dy^2 */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DoubleDPDy (
    amrex::Array4<amrex::Real> const& F, 
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
        return (F(i,j+1,k) - 2.*F(i,j,k) + F(i,j-1,k)) / (dx[1]*dx[1]);  
               
}

/**
  * Perform double derivative (d^2)P/dz^2 */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DoubleDPDz (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const z, amrex::Real const z_hi, amrex::Real const z_lo, 
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx
    ) {
        
    if (FE_lo[2] > z_lo && FE_lo[2] <= z) { //FE lower boundary
      
        if(P_BC_flag_lo == 0){
            Real F_lo = 0.0;
            return 4.*(2.*F_lo - 3.*F(i,j,k) + F(i,j,k+1))/3./dx[2]/dx[2];//2nd Order

        } else if (P_BC_flag_lo == 1){

            Real F_lo = F(i,j,k)/(1 + dx[2]/2/lambda);
            return (-dx[2]*F_lo/lambda - F(i,j,k) + F(i,j,k+1))/dx[2]/dx[2];//dPdz = P_lo/lambda;

            // Real F_lo = (9. * F(i,j,k) - F(i,j,k+1)) / (3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return  (-dx[2]*F_lo/lambda - F(i,j,k) + F(i,j,k+1))/dx[2]/dx[2];// dPdz = P_lo/lambda;

        } else if (P_BC_flag_lo == 2){
            return ( - F(i,j,k) + F(i,j,k+1))/dx[2]/dx[2];//dPdz = 0.

        } else {
            amrex::Abort("Wrong flag of the lower polarization boundary condition!!");
            return 0.0;
        }     

    } else if (z_hi > FE_hi[2] && z <= FE_hi[2] ){ // FE higher boundary

        if(P_BC_flag_hi == 0){
            Real F_hi = 0.0;
            return 4.*(2.*F_hi - 3.*F(i,j,k) + F(i,j,k-1))/3./dx[2]/dx[2];//2nd Order

        } else if (P_BC_flag_hi == 1){
            
            Real F_hi = F(i,j,k)/(1 - dx[2]/2/lambda);
            return (dx[2]*F_hi/lambda - F(i,j,k) + F(i,j,k-1))/dx[2]/dx[2];//dPdz = P_hi/lambda;

            // Real F_hi = (9. * F(i,j,k) - F(i,j,k-1)) / ( - 3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return (dx[2]*F_hi/lambda - F(i,j,k) + F(i,j,k-1))/dx[2]/dx[2]; // dPdz = P_hi/lambda;

        } else if (P_BC_flag_hi == 2){
            return ( - F(i,j,k) + F(i,j,k-1))/dx[2]/dx[2];//dPdz = 0.

        } else {
            amrex::Abort("Wrong flag of the higher polarization boundary condition!!");
            return 0.0;
        }
                  
    } else if (z_hi <= FE_hi[2] && z_lo >= FE_lo[2]) { // inside FE
        return (F(i,j,k+1) - 2.*F(i,j,k) + F(i,j,k-1)) / (dx[2]*dx[2]);  

    } else {
        return 0.0;
    }
               
}
