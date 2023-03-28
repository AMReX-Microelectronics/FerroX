/*
 * This file is part of FerroX.
 *
 * Contributor: Prabhat Kumar
 *
 */
#include <FerroXUtil.H>

using namespace amrex;


void FerroX_Util::Contains_sc(MultiFab& MaterialMask, bool& contains_SC)
{

	int has_SC = 0;

        for ( MFIter mfi(MaterialMask, TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

            const Box& bx = mfi.validbox();
            const auto lo = amrex::lbound(bx);
            const auto hi = amrex::ubound(bx);

            const Array4<Real>& mask = MaterialMask.array(mfi);

            for (auto k = lo.z; k <= hi.z; ++k) {
            for (auto j = lo.y; j <= hi.y; ++j) {
            for (auto i = lo.x; i <= hi.x; ++i) {
                  if (mask(i,j,k) >= 2.0) {
                          has_SC = 1;
                  }
            }
            }
            }

        } // end MFIter

       // parallel reduce max has_SC
       ParallelDescriptor::ReduceIntMax(has_SC);
 
       if(has_SC == 1) contains_SC = true;
}
