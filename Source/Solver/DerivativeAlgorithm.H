#include "FerroX.H"

using namespace FerroX;

/**
  * Perform first derivative dphi/dz */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DphiDz (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const z, amrex::Real const z_hi, amrex::Real const z_lo, 
    int const i, int const j, int const k,  amrex::GpuArray<amrex::Real, 3> dx, 
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>const& prob_lo,
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>const& prob_hi) {

    if (z_lo < prob_lo[2]){ // bottom metal
        return (-4.*F(i,j,k-1) + 3.*F(i,j,k) + F(i,j,k+1)) / (3. * dx[2]);
    } else if (z_hi > prob_hi[2]){ // top metal
        return (4.*F(i,j,k+1) - 3.*F(i,j,k) - F(i,j,k-1)) / (3. * dx[2]);
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
  * Perform first derivative dP/dx */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DPDx (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const x, amrex::Real const x_hi, amrex::Real const x_lo, 
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx
) {

    if (FE_lo[0] > x_lo && FE_lo[0] <= x) { //FE lower boundary
      
        if(P_BC_flag_lo[0] == 0){
            Real F_lo = 0.0;
            return (-4.*F_lo + 3.*F(i,j,k) + F(i+1,j,k))/(3.*dx[0]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i+1,j,k)

        } else if (P_BC_flag_lo[0] == 1){

            Real F_lo = F(i,j,k)/(1 + dx[0]/2/lambda);
            return (dx[0]*F_lo/lambda - F(i,j,k) + F(i+1,j,k))/(2.*dx[0]); // dP/dz = P_lo/lambda;

            // Real F_lo = (9. * F(i,j,k) - F(i,j,k+1)) / (3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return  -(dx[2]*F_lo/lambda - F(i,j,k) + F(i,j,k+1))/(2.*dx[2]);// dP/dz = P_lo/lambda;

        } else if (P_BC_flag_lo[0] == 2){
            return ( - F(i,j,k) + F(i+1,j,k))/(2.*dx[0]); //dPdx = 0.

        } else {
            amrex::Abort("Wrong flag of the lower x polarization boundary condition!!");
            return 0.0;
        }     

    } else if (x_hi > FE_hi[0] && x <= FE_hi[0] ){ // FE higher boundary

        if(P_BC_flag_hi[0] == 0){
            Real F_hi = 0.0;
            return (4.*F_hi - 3.*F(i,j,k) - F(i-1,j,k))/(3.*dx[0]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i-1,j,k)

        } else if (P_BC_flag_hi[0] == 1){
            
            Real F_hi = F(i,j,k)/(1 - dx[0]/2/lambda);
            return (dx[0]*F_hi/lambda + F(i,j,k) - F(i-1,j,k))/(2.*dx[0]);//dPdz = P_hi/lambda;

            // Real F_hi = (9. * F(i,j,k) - F(i,j,k-1)) / ( - 3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return  -(dx[2]*F_hi/lambda + F(i,j,k) - F(i,j,k-1))/(2.*dx[2]);//dPdz = P_hi/lambda;

        } else if (P_BC_flag_hi[0] == 2){
            return (F(i,j,k) - F(i-1,j,k))/(2.*dx[0]); //dPdx = 0.

        } else {
            amrex::Abort("Wrong flag of the higher x polarization boundary condition!!");
            return 0.0;
        }
                  
    } else if (x_hi <= FE_hi[0] && x_lo >= FE_lo[0]) { // inside FE
        return (F(i+1,j,k) - F(i-1,j,k))/(2.*dx[0]);

    } else {
        return 0.0;
    }
 }

/**
  * Perform first derivative dP/dy */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DPDy (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const y, amrex::Real const y_hi, amrex::Real const y_lo, 
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx
) {

    if (FE_lo[1] > y_lo && FE_lo[1] <= y) { //FE lower boundary
      
        if(P_BC_flag_lo[1] == 0){
            Real F_lo = 0.0;
            return (-4.*F_lo + 3.*F(i,j,k) + F(i,j+1,k))/(3.*dx[1]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k+1)

        } else if (P_BC_flag_lo[1] == 1){

            Real F_lo = F(i,j,k)/(1 + dx[1]/2/lambda);
            return (dx[1]*F_lo/lambda - F(i,j,k) + F(i,j+1,k))/(2.*dx[1]); // dP/dz = P_lo/lambda;

            // Real F_lo = (9. * F(i,j,k) - F(i,j,k+1)) / (3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return  -(dx[2]*F_lo/lambda - F(i,j,k) + F(i,j,k+1))/(2.*dx[2]);// dP/dz = P_lo/lambda;

        } else if (P_BC_flag_lo[1] == 2){
            return ( - F(i,j,k) + F(i,j+1,k))/(2.*dx[1]); //dPdy = 0.

        } else {
            amrex::Abort("Wrong flag of the lower polarization boundary condition!!");
            return 0.0;
        }     

    } else if (y_hi > FE_hi[1] && y <= FE_hi[1] ){ // FE higher boundary

        if(P_BC_flag_hi[1] == 0){
            Real F_hi = 0.0;
            return (4.*F_hi - 3.*F(i,j,k) - F(i,j-1,k))/(3.*dx[1]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k-1)

        } else if (P_BC_flag_hi[1] == 1){
            
            Real F_hi = F(i,j,k)/(1 - dx[1]/2/lambda);
            return (dx[1]*F_hi/lambda + F(i,j,k) - F(i,j-1,k))/(2.*dx[1]);//dPdz = P_hi/lambda;

            // Real F_hi = (9. * F(i,j,k) - F(i,j,k-1)) / ( - 3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return  -(dx[2]*F_hi/lambda + F(i,j,k) - F(i,j,k-1))/(2.*dx[2]);//dPdz = P_hi/lambda;

        } else if (P_BC_flag_hi[1] == 2){
            return (F(i,j,k) - F(i,j-1,k))/(2.*dx[1]); //dPdy = 0.

        } else {
            amrex::Abort("Wrong flag of the higher y polarization boundary condition!!");
            return 0.0;
        }
                  
    } else if (y_hi <= FE_hi[1] && y_lo >= FE_lo[1]) { // inside FE
        return (F(i,j+1,k) - F(i,j-1,k))/(2.*dx[1]);

    } else {
        return 0.0;
    }
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
      
        if(P_BC_flag_lo[2] == 0){
            Real F_lo = 0.0;
            return (-4.*F_lo + 3.*F(i,j,k) + F(i,j,k+1))/(3.*dx[2]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k+1)

        } else if (P_BC_flag_lo[2] == 1){

            Real F_lo = F(i,j,k)/(1 + dx[2]/2/lambda);
            return (dx[2]*F_lo/lambda - F(i,j,k) + F(i,j,k+1))/(2.*dx[2]); // dP/dz = P_lo/lambda;

            // Real F_lo = (9. * F(i,j,k) - F(i,j,k+1)) / (3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return  -(dx[2]*F_lo/lambda - F(i,j,k) + F(i,j,k+1))/(2.*dx[2]);// dP/dz = P_lo/lambda;

        } else if (P_BC_flag_lo[2] == 2){
            return ( - F(i,j,k) + F(i,j,k+1))/(2.*dx[2]); //dPdz = 0.

        } else {
            amrex::Abort("Wrong flag of the lower polarization boundary condition!!");
            return 0.0;
        }     

    } else if (z_hi > FE_hi[2] && z <= FE_hi[2] ){ // FE higher boundary

        if(P_BC_flag_hi[2] == 0){
            Real F_hi = 0.0;
            return (4.*F_hi - 3.*F(i,j,k) - F(i,j,k-1))/(3.*dx[2]);//2nd order using three point stencil using 0, pOld(i,j,k), and pOld(i,j,k-1)

        } else if (P_BC_flag_hi[2] == 1){
            
            Real F_hi = F(i,j,k)/(1 - dx[2]/2/lambda);
            return (dx[2]*F_hi/lambda + F(i,j,k) - F(i,j,k-1))/(2.*dx[2]);//dPdz = P_hi/lambda;

            // Real F_hi = (9. * F(i,j,k) - F(i,j,k-1)) / ( - 3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return  -(dx[2]*F_hi/lambda + F(i,j,k) - F(i,j,k-1))/(2.*dx[2]);//dPdz = P_hi/lambda;

        } else if (P_BC_flag_hi[2] == 2){
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
//
///**
//  * Perform double derivative (d^2)P/dx^2 */
// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
// static amrex::Real DoubleDPDx (
//    amrex::Array4<amrex::Real> const& F, 
//    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
//        return (F(i+1,j,k) - 2.*F(i,j,k) + F(i-1,j,k)) / (dx[0]*dx[0]);  
//               
//}
//
///**
//  * Perform double derivative (d^2)P/dy^2 */
// AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
// static amrex::Real DoubleDPDy (
//    amrex::Array4<amrex::Real> const& F, 
//    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx) {
//        return (F(i,j+1,k) - 2.*F(i,j,k) + F(i,j-1,k)) / (dx[1]*dx[1]);  
//               
//}
//

/**
  * Perform double derivative (d^2)P/dx^2 */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DoubleDPDx (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const x, amrex::Real const x_hi, amrex::Real const x_lo, 
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx
    ) {
        
    if (FE_lo[0] > x_lo && FE_lo[0] <= x) { //FE lower boundary
      
        if(P_BC_flag_lo[0] == 0){
            Real F_lo = 0.0;
            return 4.*(2.*F_lo - 3.*F(i,j,k) + F(i+1,j,k))/3./dx[0]/dx[0];//2nd Order

        } else if (P_BC_flag_lo[0] == 1){

            Real F_lo = F(i,j,k)/(1 + dx[0]/2/lambda);
            return (-dx[0]*F_lo/lambda - F(i,j,k) + F(i+1,j,k))/dx[0]/dx[0];//dPdx = P_lo/lambda;

            // Real F_lo = (9. * F(i,j,k) - F(i,j,k+1)) / (3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return  (-dx[2]*F_lo/lambda - F(i,j,k) + F(i,j,k+1))/dx[2]/dx[2];// dPdz = P_lo/lambda;

        } else if (P_BC_flag_lo[0] == 2){
            return ( - F(i,j,k) + F(i+1,j,k))/dx[0]/dx[0];//dPdx = 0.

        } else {
            amrex::Abort("Wrong flag of the lower polarization boundary condition!!");
            return 0.0;
        }     

    } else if (x_hi > FE_hi[0] && x <= FE_hi[0] ){ // FE higher boundary

        if(P_BC_flag_hi[0] == 0){
            Real F_hi = 0.0;
            return 4.*(2.*F_hi - 3.*F(i,j,k) + F(i-1,j,k))/3./dx[0]/dx[0];//2nd Order

        } else if (P_BC_flag_hi[0] == 1){
            
            Real F_hi = F(i,j,k)/(1 - dx[0]/2/lambda);
            return (dx[0]*F_hi/lambda - F(i,j,k) + F(i-1,j,k))/dx[0]/dx[0];//dPdx = P_hi/lambda;

            // Real F_hi = (9. * F(i,j,k) - F(i,j,k-1)) / ( - 3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return (dx[2]*F_hi/lambda - F(i,j,k) + F(i,j,k-1))/dx[2]/dx[2]; // dPdz = P_hi/lambda;

        } else if (P_BC_flag_hi[0] == 2){
            return ( - F(i,j,k) + F(i-1,j,k))/dx[0]/dx[0];//dPdx = 0.

        } else {
            amrex::Abort("Wrong flag of the higher polarization boundary condition!!");
            return 0.0;
        }
                  
    } else if (x_hi <= FE_hi[0] && x_lo >= FE_lo[0]) { // inside FE
        return (F(i+1,j,k) - 2.*F(i,j,k) + F(i-1,j,k)) / (dx[0]*dx[0]);  

    } else {
        return 0.0;
    }
               
}

/**
  * Perform double derivative (d^2)P/dy^2 */
 AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
 static amrex::Real DoubleDPDy (
    amrex::Array4<amrex::Real> const& F,
    amrex::Real const y, amrex::Real const y_hi, amrex::Real const y_lo, 
    int const i, int const j, int const k, amrex::GpuArray<amrex::Real, 3> dx
    ) {
        
    if (FE_lo[1] > y_lo && FE_lo[1] <= y) { //FE lower boundary
      
        if(P_BC_flag_lo[1] == 0){
            Real F_lo = 0.0;
            return 4.*(2.*F_lo - 3.*F(i,j,k) + F(i,j+1,k))/3./dx[1]/dx[1];//2nd Order

        } else if (P_BC_flag_lo[1] == 1){

            Real F_lo = F(i,j,k)/(1 + dx[1]/2/lambda);
            return (-dx[1]*F_lo/lambda - F(i,j,k) + F(i,j+1,k))/dx[1]/dx[1];//dPdz = P_lo/lambda;

            // Real F_lo = (9. * F(i,j,k) - F(i,j,k+1)) / (3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return  (-dx[2]*F_lo/lambda - F(i,j,k) + F(i,j,k+1))/dx[2]/dx[2];// dPdz = P_lo/lambda;

        } else if (P_BC_flag_lo[1] == 2){
            return ( - F(i,j,k) + F(i,j+1,k))/dx[1]/dx[1];//dPdy = 0.

        } else {
            amrex::Abort("Wrong flag of the lower polarization boundary condition!!");
            return 0.0;
        }     

    } else if (y_hi > FE_hi[1] && y <= FE_hi[1] ){ // FE higher boundary

        if(P_BC_flag_hi[1] == 0){
            Real F_hi = 0.0;
            return 4.*(2.*F_hi - 3.*F(i,j,k) + F(i,j-1,k))/3./dx[1]/dx[1];//2nd Order

        } else if (P_BC_flag_hi[1] == 1){
            
            Real F_hi = F(i,j,k)/(1 - dx[1]/2/lambda);
            return (dx[1]*F_hi/lambda - F(i,j,k) + F(i,j-1,k))/dx[1]/dx[1];//dPdy = P_hi/lambda;

            // Real F_hi = (9. * F(i,j,k) - F(i,j,k-1)) / ( - 3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return (dx[2]*F_hi/lambda - F(i,j,k) + F(i,j,k-1))/dx[2]/dx[2]; // dPdz = P_hi/lambda;

        } else if (P_BC_flag_hi[1] == 2){
            return ( - F(i,j,k) + F(i,j-1,k))/dx[1]/dx[1];//dPdy = 0.

        } else {
            amrex::Abort("Wrong flag of the higher polarization boundary condition!!");
            return 0.0;
        }
                  
    } else if (y_hi <= FE_hi[1] && y_lo >= FE_lo[1]) { // inside FE
        return (F(i,j+1,k) - 2.*F(i,j,k) + F(i,j-1,k)) / (dx[1]*dx[1]);  

    } else {
        return 0.0;
    }
               
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
      
        if(P_BC_flag_lo[2] == 0){
            Real F_lo = 0.0;
            return 4.*(2.*F_lo - 3.*F(i,j,k) + F(i,j,k+1))/3./dx[2]/dx[2];//2nd Order

        } else if (P_BC_flag_lo[2] == 1){

            Real F_lo = F(i,j,k)/(1 + dx[2]/2/lambda);
            return (-dx[2]*F_lo/lambda - F(i,j,k) + F(i,j,k+1))/dx[2]/dx[2];//dPdz = P_lo/lambda;

            // Real F_lo = (9. * F(i,j,k) - F(i,j,k+1)) / (3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return  (-dx[2]*F_lo/lambda - F(i,j,k) + F(i,j,k+1))/dx[2]/dx[2];// dPdz = P_lo/lambda;

        } else if (P_BC_flag_lo[2] == 2){
            return ( - F(i,j,k) + F(i,j,k+1))/dx[2]/dx[2];//dPdz = 0.

        } else {
            amrex::Abort("Wrong flag of the lower polarization boundary condition!!");
            return 0.0;
        }     

    } else if (z_hi > FE_hi[2] && z <= FE_hi[2] ){ // FE higher boundary

        if(P_BC_flag_hi[2] == 0){
            Real F_hi = 0.0;
            return 4.*(2.*F_hi - 3.*F(i,j,k) + F(i,j,k-1))/3./dx[2]/dx[2];//2nd Order

        } else if (P_BC_flag_hi[2] == 1){
            
            Real F_hi = F(i,j,k)/(1 - dx[2]/2/lambda);
            return (dx[2]*F_hi/lambda - F(i,j,k) + F(i,j,k-1))/dx[2]/dx[2];//dPdz = P_hi/lambda;

            // Real F_hi = (9. * F(i,j,k) - F(i,j,k-1)) / ( - 3. * dx[2] / lambda + 8.); // derived with 2nd order one-sided 1st derivative 
            // return (dx[2]*F_hi/lambda - F(i,j,k) + F(i,j,k-1))/dx[2]/dx[2]; // dPdz = P_hi/lambda;

        } else if (P_BC_flag_hi[2] == 2){
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
