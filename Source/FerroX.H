/* Contributors: Prabhat Kumar, Saurabh Sawant
 *
 */
#ifndef FERROX_H_
#define FERROX_H_

#include "Input/GeometryProperties/GeometryProperties_fwd.H"
#include "Input/BoundaryConditions/BoundaryConditions_fwd.H"
#include "Utils/SelectWarpXUtils/WarnManager_fwd.H"


#include <AMReX.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_REAL.H>
#include <AMReX_MLLinOp.H>
#include <AMReX_Geometry.H>
#include "FerroX_namespace.H"

using namespace FerroX;
using namespace amrex;

enum class WarnPriority
{
    /** Low priority warning:
     * essentially an informative message
     */
    low, 
    /** Medium priority warning:
     * a bug or a performance issue may affect the simulation
     */
    medium,
    /** High priority warning:
     * a very serious bug or performance issue
     * almost certainly affects the simulation
     */
    high
};

class c_FerroX
{

public:

    c_FerroX (); //constructor
    ~c_FerroX (); //destructor


    /********************* FUNCTIONS/PARAMETERS TAKEN FROM WARPX ****************************/

    static c_FerroX& GetInstance ();
    static void ResetInstance ();

    /**
     * \brief This function is borrowed from WarpX. It records a warning message.
     * RecordWarning is thread safe: it can be used within OpenMP parallel loops.
     *
     * @param[in] topic a string to identify the topic of the warning 
     * (e.g., "parallelization", "pbc", "particles"...)
     * @param[in] text the text of the warning message
     * @param[in] priority priority of the warning message ("medium" by default)
     */
    void RecordWarning(
        std::string topic,
        std::string text,
        WarnPriority priority = WarnPriority::medium);

    /**
     * \brief This function prints all the warning messages collected on the present MPI rank
     * (i.e., this is not a collective call). This function is mainly intended for debug purposes.
     *
     * @param[in] when a string to mark when the warnings are printed out (it appears in the warning list)
     */
    void PrintLocalWarnings(const std::string& when);

    /**
     * \brief This function prints all the warning messages collected by all the MPI ranks
     * (i.e., this is a collective call). Only the I/O rank prints the message.
     *
     * @param[in] when a string to mark when the warnings are printed out (it appears in the warning list)
     */
    void PrintGlobalWarnings(const std::string& when);


    std::unique_ptr<Utils::WarnManager> m_p_warn_manager; //Warning manager: used to record and print error msgs.
    bool m_always_warn_immediately = false; //flag to emit control emitting warning as soon as it is recorded

    /****************************************************************************************/ 


    void ReadData ();
    void InitData ();

    static bool do_device_synchronize;

    c_GeometryProperties& get_GeometryProperties () { return *m_pGeometryProperties;}
    c_BoundaryConditions& get_BoundaryConditions () { return *m_pBoundaryConditions;}
    const amrex::Real get_time() { return m_time_instant;}
    const amrex::Real set_time(int n) { m_time_instant = n*m_timestep; return m_time_instant;}

private:

    static c_FerroX* m_instance;

    amrex::Real m_timestep;
    int m_total_steps;

    amrex::Real m_time_instant = 0;

    std::unique_ptr<c_GeometryProperties> m_pGeometryProperties;
    std::unique_ptr<c_BoundaryConditions> m_pBoundaryConditions;

};


void main_main (c_FerroX& rFerroX);
void InitializeFerroXNamespace (const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
                                const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_hi);

#endif
