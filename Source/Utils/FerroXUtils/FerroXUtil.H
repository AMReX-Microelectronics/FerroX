/*
 * This file is part of FerroX.
 *
 * Contributor: Prabhat Kumar
 *
 */
#include <AMReX_REAL.H>
#include <AMReX_Array.H>
#include <AMReX_Vector.H>
#include <AMReX_MultiFab.H>
#include <AMReX_Parser.H>


#include <ctype.h>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <string>
#include <vector>
#include <any>

using namespace amrex;


namespace FerroX_MFab_Util
{
void AverageFaceCenteredMultiFabToCellCenters(std::array< amrex::MultiFab,
                                            AMREX_SPACEDIM >& fc_arr,
                                            amrex::MultiFab& cc_arr);

}
