#include "CoordinateTransformation.H"

/*
 P_i^L = R_ij*P_j, where 
 R_ij = [cos(a)cos(t)-cos(b)sin(a)sin(t) 	sin(a)cos(t)+cos(b)cos(a)sin(t) 	sin(b)sin(t)]
        [-cos(b)cos(t)sin(a)-cos(a)sin(t) 	cos(b)cos(a)cos(t)-sin(a)sin(t) 	sin(b)cos(t)]
        [sin(a)sin(b) 				-cos(a)sin(b) 				cos(b)	    ]
 */
void transform_global_to_local(Array<MultiFab, AMREX_SPACEDIM> &src,
                               Array<MultiFab, AMREX_SPACEDIM> &dst)
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

	amrex::Real R_11 = cos(angle_alpha)*cos(angle_theta) - cos(angle_beta)*sin(angle_alpha)*sin(angle_theta);  
	amrex::Real R_12 = sin(angle_alpha)*cos(angle_theta) + cos(angle_beta)*cos(angle_alpha)*sin(angle_theta);  
	amrex::Real R_13 = sin(angle_beta)*sin(angle_theta);  
	amrex::Real R_21 = -cos(angle_beta)*cos(angle_theta)*sin(angle_alpha) - cos(angle_alpha)*sin(angle_theta);  
	amrex::Real R_22 = cos(angle_beta)*cos(angle_alpha)*cos(angle_theta) - sin(angle_alpha)*sin(angle_theta);  
	amrex::Real R_23 = sin(angle_beta)*cos(angle_theta);  
	amrex::Real R_31 = sin(angle_alpha)*sin(angle_beta);  
	amrex::Real R_32 = -cos(angle_alpha)*sin(angle_beta);  
	amrex::Real R_33 = cos(angle_beta);  

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
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
                               Array<MultiFab, AMREX_SPACEDIM> &dst)
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

	amrex::Real iR_11 = cos(angle_alpha)*cos(angle_theta) - cos(angle_beta)*sin(angle_alpha)*sin(angle_theta);  
	amrex::Real iR_12 = -sin(angle_alpha)*cos(angle_beta)*cos(angle_theta) - cos(angle_alpha)*sin(angle_theta);  
	amrex::Real iR_13 = sin(angle_beta)*sin(angle_alpha);  
	amrex::Real iR_21 = cos(angle_theta)*sin(angle_alpha) + cos(angle_alpha)*sin(angle_theta)*cos(angle_beta);  
	amrex::Real iR_22 = cos(angle_beta)*cos(angle_alpha)*cos(angle_theta) - sin(angle_alpha)*sin(angle_theta);  
	amrex::Real iR_23 = -sin(angle_beta)*cos(angle_alpha);  
	amrex::Real iR_31 = sin(angle_theta)*sin(angle_beta);  
	amrex::Real iR_32 = cos(angle_theta)*sin(angle_beta);  
	amrex::Real iR_33 = cos(angle_beta);  

        amrex::ParallelFor( bx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
             dst_x(i,j,k) = iR_11*src_x(i,j,k) + iR_12*src_y(i,j,k) + iR_13*src_z(i,j,k);
             dst_y(i,j,k) = iR_21*src_x(i,j,k) + iR_22*src_y(i,j,k) + iR_23*src_z(i,j,k);
             dst_z(i,j,k) = iR_31*src_x(i,j,k) + iR_32*src_y(i,j,k) + iR_33*src_z(i,j,k);
        });
    }
}
