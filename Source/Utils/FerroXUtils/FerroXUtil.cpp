/*
 * This file is part of FerroX.
 *
 * Contributor: Prabhat Kumar
 *
 */
#include <FerroXUtil.H>

using namespace amrex;


void FerroX_MFab_Util::AverageFaceCenteredMultiFabToCellCenters(std::array< amrex::MultiFab,
                                            AMREX_SPACEDIM >& fc_arr,
                                            amrex::MultiFab& cc_arr)
{

    for (MFIter mfi(cc_arr); mfi.isValid(); ++mfi)
    {
        const Array4<Real> & cc = cc_arr.array(mfi);
        const Array4<Real> & facex = fc_arr[0].array(mfi);
        const Array4<Real> & facey = fc_arr[1].array(mfi);
        const Array4<Real> & facez = fc_arr[2].array(mfi);

	const Box& bx = mfi.validbox();
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cc(i,j,k) = 0.5*(facex(i,j,k)+facex(i+1,j,k));
            //cc(i,j,k) = 0.5*(facey(i,j,k)+facey(i,j+1,k));
            //cc(i,j,k) = 0.5*(facez(i,j,k)+facez(i,j,k+1));
        });
    }

}
