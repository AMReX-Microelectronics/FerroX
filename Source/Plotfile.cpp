#include "FerroX.H"
#include "AMReX_PlotFileUtil.H"
#include "Input/GeometryProperties/GeometryProperties.H"

void WritePlotfile(c_FerroX& rFerroX,
                   MultiFab& PoissonPhi,
                   MultiFab& PoissonRHS,
                   Array< MultiFab, AMREX_SPACEDIM>& P_old,
                   Array< MultiFab, AMREX_SPACEDIM>& E,
                   MultiFab& hole_den,
                   MultiFab& e_den,
                   MultiFab& charge_den,
                   MultiFab& beta_cc,
                   MultiFab& MaterialMask,
                   MultiFab& tphaseMask,
                   MultiFab& angle_alpha,
                   MultiFab& angle_beta,
                   MultiFab& angle_theta,
                   MultiFab& Phidiff,
                   const Geometry& geom,
                   const Real& time,
                   const int& plt_step)
{
    // timer for profiling
    BL_PROFILE_VAR("WritePlotfile()",WritePlotfile);

    BoxArray ba = PoissonPhi.boxArray();
    DistributionMapping dm = PoissonPhi.DistributionMap();

    const std::string& pltfile = amrex::Concatenate("plt",plt_step,8);

    Vector<std::string> var_names;

    //Px, Py, Pz
    int nvar = 3;

    var_names.push_back("Px");
    var_names.push_back("Py");
    var_names.push_back("Pz");

    if (plot_Phi) {
        ++nvar;
        var_names.push_back("Phi");
    }

    if (plot_PoissonRHS) {
        ++nvar;
        var_names.push_back("PoissonRHS");
    }

    if (plot_E) {
        nvar += 3;
        var_names.push_back("Ex");
        var_names.push_back("Ey");
        var_names.push_back("Ez");
    }

    if (plot_holes) {
        ++nvar;
        var_names.push_back("holes");
    }

    if (plot_electrons) {
        ++nvar;
        var_names.push_back("electrons");
    }

    if (plot_charge) {
        ++nvar;
        var_names.push_back("charge");
    }

    if (plot_epsilon) {
        ++nvar;
        var_names.push_back("epsilon");
    }

    if (plot_mask) {
        ++nvar;
        var_names.push_back("mask");
    }

    if (plot_tphase) {
        ++nvar;
        var_names.push_back("tphase");
    }

    if (plot_alpha) {
        ++nvar;
        var_names.push_back("alpha");
    }

    if (plot_beta) {
        ++nvar;
        var_names.push_back("beta");
    }

    if (plot_theta) {
        ++nvar;
        var_names.push_back("theta");
    }

    if (plot_PhiDiff) {
        ++nvar;
        var_names.push_back("PhiDiff");
    }



    auto& rGprop = rFerroX.get_GeometryProperties();
#ifdef AMREX_USE_EB
    MultiFab Plt(ba, dm, nvar, 0,  MFInfo(), *rGprop.pEB->p_factory_union);
#else    
    MultiFab Plt(ba, dm, nvar, 0);
#endif

    int counter = 0;

    MultiFab::Copy(Plt, P_old[0], 0, counter++, 1, 0);
    MultiFab::Copy(Plt, P_old[1], 0, counter++, 1, 0);
    MultiFab::Copy(Plt, P_old[2], 0, counter++, 1, 0);  

    if (plot_Phi) {
        MultiFab::Copy(Plt, PoissonPhi, 0, counter++, 1, 0);
    }

    if (plot_PoissonRHS) {
        MultiFab::Copy(Plt, PoissonRHS, 0, counter++, 1, 0);
    }

    if (plot_E) {
        MultiFab::Copy(Plt, E[0], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, E[1], 0, counter++, 1, 0);
        MultiFab::Copy(Plt, E[2], 0, counter++, 1, 0);  
    }

    if (plot_holes) {
        MultiFab::Copy(Plt, hole_den, 0, counter++, 1, 0);
    }

    if (plot_electrons) {
        MultiFab::Copy(Plt, e_den, 0, counter++, 1, 0);
    }

    if (plot_charge) {
        MultiFab::Copy(Plt, charge_den, 0, counter++, 1, 0);
    }

    if (plot_epsilon) {
        MultiFab::Copy(Plt, beta_cc, 0, counter++, 1, 0);
    }

    if (plot_mask) {
        MultiFab::Copy(Plt, MaterialMask, 0, counter++, 1, 0);
    }

    if (plot_tphase) {
        MultiFab::Copy(Plt, tphaseMask, 0, counter++, 1, 0);
    }

    if (plot_alpha) {
        MultiFab::Copy(Plt, angle_alpha, 0, counter++, 1, 0);
    }

    if (plot_beta) {
        MultiFab::Copy(Plt, angle_beta, 0, counter++, 1, 0);
    }

    if (plot_theta) {
        MultiFab::Copy(Plt, angle_theta, 0, counter++, 1, 0);
    }

    if (plot_PhiDiff) {
        MultiFab::Copy(Plt, Phidiff, 0, counter++, 1, 0);
    }

    WriteSingleLevelPlotfile(pltfile, Plt, var_names, geom, time, plt_step);
}