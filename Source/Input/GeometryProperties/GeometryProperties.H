/*
 * This file is taken from eXstatic.
 *
 * Contributor: Saurabh S. Sawant
 *
 */
#ifndef GEOMETRY_PROPERTIES_H_
#define GEOMETRY_PROPERTIES_H_

#include "GeometryProperties_fwd.H"
#ifdef AMREX_USE_EB
#include "EmbeddedBoundaries.H"
#endif
#include <AMReX_REAL.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Vector.H>
#include <AMReX_IntVect.H>
#include <AMReX_BoxArray.H>
#include <AMReX_Geometry.H>

#include <AMReX.H>
#ifdef AMREX_USE_EB
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EBSupport.H>
#endif


#include <string>

class 
c_GeometryProperties
{
public: 
    c_GeometryProperties();
    ~c_GeometryProperties();

    amrex::GpuArray<int, AMREX_SPACEDIM>  n_cell;     // number of cells in each dimension
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>  prob_lo; // physical lo coordinate
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>  prob_hi; // physical hi coordinate
    amrex::IntVect max_grid_size;
    amrex::IntVect blocking_factor;
    std::array<int, AMREX_SPACEDIM> is_periodic;
    amrex::CoordSys::CoordType coord_sys;

    amrex::BoxArray ba; // a list of boxes that cover the domain
    amrex::Geometry geom;
    amrex::DistributionMapping dm;

    int embedded_boundary_flag;
#ifdef AMREX_USE_EB
    std::unique_ptr<c_EmbeddedBoundaries> pEB;
#endif
    void ReadData();
    void InitData();

private:
    void ParseBasicDomainInput();
    void InitializeBoxArrayAndDistributionMap ();

};

#endif
