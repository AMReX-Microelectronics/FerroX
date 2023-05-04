#include "CoordinateTransformation.H"

/*
 P_i^L = R_ij*P_j, where 
 R_ij = [cos(a)cos(t)-cos(b)sin(a)sin(t) 	sin(a)cos(t)+cos(b)cos(a)sin(t) 	sin(b)sin(t)]
        [-cos(b)cos(t)sin(a)-cos(a)sin(t) 	cos(b)cos(a)cos(t)-sin(a)sin(t) 	sin(b)cos(t)]
        [sin(a)sin(b) 				-cos(a)sin(b) 				cos(b)	    ]
 */
void transform_global_to_local(Array<MultiFab, AMREX_SPACEDIM> &src,
                               Array<MultiFab, AMREX_SPACEDIM> &dst,
			       MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta)
{
    for (MFIter mfi(src[0]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

	const Array4<Real> &src_x = src[0].array(mfi);
        const Array4<Real> &src_y = src[1].array(mfi);
        const Array4<Real> &src_z = src[2].array(mfi);

	const Array4<Real> &dst_x = dst[0].array(mfi);
        const Array4<Real> &dst_y = dst[1].array(mfi);
        const Array4<Real> &dst_z = dst[2].array(mfi);

	const Array4<Real> &alpha_arr = angle_alpha.array(mfi);
        const Array4<Real> &beta_arr = angle_beta.array(mfi);
        const Array4<Real> &theta_arr = angle_theta.array(mfi);


        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

	     //Convert Euler angles from degrees to radians 
	     amrex::Real alpha_rad = 0.0174533*alpha_arr(i,j,k);
	     amrex::Real beta_rad = 0.0174533*beta_arr(i,j,k);
	     amrex::Real theta_rad = 0.0174533*theta_arr(i,j,k);

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
             dst_x(i,j,k) = R_11*src_x(i,j,k) + R_12*src_y(i,j,k) + R_13*src_z(i,j,k);
             dst_y(i,j,k) = R_21*src_x(i,j,k) + R_22*src_y(i,j,k) + R_23*src_z(i,j,k);
             dst_z(i,j,k) = R_31*src_x(i,j,k) + R_32*src_y(i,j,k) + R_33*src_z(i,j,k);
        });
    }
}

/*
 P_j = inv(R_ij)*P_i^L, where 
 inv(R_ij) = [-cos(b)sin(a)sin(t)+cos(a)cos(t) 		-sin(a)cos(b)cos(t)- cos(a)sin(t)  	sin(a)sin(b) ]
             [cos(t)sin(a)+cos(a)sin(t)cos(b) 		-sin(a)sin(t)+cos(b)cos(a)cos(t) 	-sin(b)cos(a)]
             [sin(t)sin(b) 				cos(t)sin(b) 				cos(b)       ]
 */
void transform_local_to_global(Array<MultiFab, AMREX_SPACEDIM> &src,
                               Array<MultiFab, AMREX_SPACEDIM> &dst,
			       MultiFab& angle_alpha, MultiFab& angle_beta, MultiFab& angle_theta)
{
    for (MFIter mfi(src[0]); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.validbox();

	const Array4<Real> &src_x = src[0].array(mfi);
        const Array4<Real> &src_y = src[1].array(mfi);
        const Array4<Real> &src_z = src[2].array(mfi);

	const Array4<Real> &dst_x = dst[0].array(mfi);
        const Array4<Real> &dst_y = dst[1].array(mfi);
        const Array4<Real> &dst_z = dst[2].array(mfi);

	const Array4<Real> &alpha_arr = angle_alpha.array(mfi);
        const Array4<Real> &beta_arr = angle_beta.array(mfi);
        const Array4<Real> &theta_arr = angle_theta.array(mfi);

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {

	     //Convert Euler angles from degrees to radians 
	     amrex::Real alpha_rad = 0.0174533*alpha_arr(i,j,k);
	     amrex::Real beta_rad = 0.0174533*beta_arr(i,j,k);
	     amrex::Real theta_rad = 0.0174533*theta_arr(i,j,k);

             amrex::Real iR_11, iR_12, iR_13, iR_21, iR_22, iR_23, iR_31, iR_32, iR_33;

             if(use_Euler_angles == 1){
	     	iR_11 = cos(alpha_rad)*cos(theta_rad) - cos(beta_rad)*sin(alpha_rad)*sin(theta_rad);  
	     	iR_12 = -sin(alpha_rad)*cos(beta_rad)*cos(theta_rad) - cos(alpha_rad)*sin(theta_rad);  
	     	iR_13 = sin(beta_rad)*sin(alpha_rad);  
	     	iR_21 = cos(theta_rad)*sin(alpha_rad) + cos(alpha_rad)*sin(theta_rad)*cos(beta_rad);  
	     	iR_22 = cos(beta_rad)*cos(alpha_rad)*cos(theta_rad) - sin(alpha_rad)*sin(theta_rad);  
	     	iR_23 = -sin(beta_rad)*cos(alpha_rad);  
	     	iR_31 = sin(theta_rad)*sin(beta_rad);  
	     	iR_32 = cos(theta_rad)*sin(beta_rad);  
	     	iR_33 = cos(beta_rad);  
             } else {
	     	iR_11 = cos(beta_rad)*cos(theta_rad);  
	     	iR_21 = sin(alpha_rad)*sin(beta_rad)*cos(theta_rad) - cos(alpha_rad)*sin(theta_rad);  
	     	iR_31 = cos(alpha_rad)*sin(beta_rad)*cos(theta_rad) + sin(alpha_rad)*sin(theta_rad);  
	     	iR_12 = cos(beta_rad)*sin(theta_rad);  
	     	iR_22 = sin(beta_rad)*sin(alpha_rad)*sin(theta_rad) + cos(alpha_rad)*cos(theta_rad);  
	     	iR_32 = cos(alpha_rad)*sin(beta_rad)*sin(theta_rad) - sin(alpha_rad)*cos(theta_rad);  
	     	iR_13 = -sin(beta_rad);  
	     	iR_23 = sin(alpha_rad)*cos(beta_rad);  
	     	iR_33 = cos(alpha_rad)*cos(beta_rad);  
             }

             dst_x(i,j,k) = iR_11*src_x(i,j,k) + iR_12*src_y(i,j,k) + iR_13*src_z(i,j,k);
             dst_y(i,j,k) = iR_21*src_x(i,j,k) + iR_22*src_y(i,j,k) + iR_23*src_z(i,j,k);
             dst_z(i,j,k) = iR_31*src_x(i,j,k) + iR_32*src_y(i,j,k) + iR_33*src_z(i,j,k);
        });
    }
}
