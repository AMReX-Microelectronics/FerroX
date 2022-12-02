#ifndef _ferrox_H_
#define _ferrox_H_

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include "FerroX_namespace.H"
#include "Code.H"

using namespace FerroX;
using namespace amrex;

void main_main (c_Code& rCode);
void InitializeFerroXNamespace ();

#endif
