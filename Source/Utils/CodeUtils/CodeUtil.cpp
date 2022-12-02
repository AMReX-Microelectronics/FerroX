#include <CodeUtil.H>
//#include "../../Code.H"
//#include "../../Input/MacroscopicProperties/MacroscopicProperties.H"
//#include "../../PostProcessor/PostProcessor.H"

using namespace amrex;

#define VFRAC_THREASHOLD 1e-4

void 
Multifab_Manipulation::InitializeMacroMultiFabUsingParser_3vars (amrex::MultiFab *macro_mf,
                                                                 amrex::ParserExecutor<3> const& macro_parser,
                                                                 amrex::Geometry& geom)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::InitializeMacroMultiFabUsingParser************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    auto dx = geom.CellSizeArray();
    //amrex::Print() << "dx: " << dx[0] << " " << dx[1] << " " << dx[2] << "\n";

    auto& real_box = geom.ProbDomain();
    //amrex::Print() << "real_box_lo: " << real_box.lo(0) << " " << real_box.lo(1) << " " << real_box.lo(2) << "\n";

    auto iv = macro_mf->ixType().toIntVect();
    //amrex::Print() << "iv: " << iv[0] << " " << iv[1] << " " << iv[2] << "\n";

    for ( amrex::MFIter mfi(*macro_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        const auto& tb = mfi.tilebox( iv, macro_mf->nGrowVect() ); /** initialize ghost cells in addition to valid cells.
                                                                       auto = amrex::Box
                                                                    */
        auto const& mf_array =  macro_mf->array(mfi); //auto = amrex::Array4<amrex::Real>

        amrex::ParallelFor (tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                amrex::Real fac_x = (1._rt - iv[0]) * dx[0] * 0.5_rt;
                amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;

                amrex::Real fac_y = (1._rt - iv[1]) * dx[1] * 0.5_rt;
                amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;

                amrex::Real fac_z = (1._rt - iv[2]) * dx[2] * 0.5_rt;
                amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;

                mf_array(i,j,k) = macro_parser(x,y,z);
        });
    }
#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************Multifab_Manipulation::InitializeMacroMultiFabUsingParser************************\n";
#endif
}


void 
Multifab_Manipulation::InitializeMacroMultiFabUsingParser_4vars (amrex::MultiFab *macro_mf,
                                                                 amrex::ParserExecutor<4> const& macro_parser,
                                                                 amrex::Geometry& geom,
                                                                 const amrex::Real t)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::InitializeMacroMultiFabUsingParser************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    auto dx = geom.CellSizeArray();

    auto& real_box = geom.ProbDomain();

    auto iv = macro_mf->ixType().toIntVect();

    for ( amrex::MFIter mfi(*macro_mf, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) {

        const auto& tb = mfi.tilebox( iv, macro_mf->nGrowVect() ); /** initialize ghost cells in addition to valid cells.
                                                                       auto = amrex::Box
                                                                    */
        auto const& mf_array =  macro_mf->array(mfi); //auto = amrex::Array4<amrex::Real>
        
        amrex::ParallelFor (tb,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                amrex::Real fac_x = (1._rt - iv[0]) * dx[0] * 0.5_rt;
                amrex::Real x = i * dx[0] + real_box.lo(0) + fac_x;

                amrex::Real fac_y = (1._rt - iv[1]) * dx[1] * 0.5_rt;
                amrex::Real y = j * dx[1] + real_box.lo(1) + fac_y;

                amrex::Real fac_z = (1._rt - iv[2]) * dx[2] * 0.5_rt;
                amrex::Real z = k * dx[2] + real_box.lo(2) + fac_z;

                mf_array(i,j,k) = macro_parser(x,y,z,t);
        });
    }
#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************Multifab_Manipulation::InitializeMacroMultiFabUsingParser************************\n";
#endif
}


void 
Multifab_Manipulation::AverageCellCenteredMultiFabToCellFaces(const amrex::MultiFab& cc_arr,
                                       std::array< amrex::MultiFab, 
                                       AMREX_SPACEDIM >& face_arr)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::AverageCellCenteredMultiFabToCellFaces()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif
    for (MFIter mfi(cc_arr, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Array4<Real const> & cc = cc_arr.array(mfi);
        AMREX_D_TERM(const Array4<Real> & facex = face_arr[0].array(mfi);,
                     const Array4<Real> & facey = face_arr[1].array(mfi);,
                     const Array4<Real> & facez = face_arr[2].array(mfi););

        AMREX_D_TERM(const Box & nodal_x = mfi.nodaltilebox(0);,
                     const Box & nodal_y = mfi.nodaltilebox(1);,
                     const Box & nodal_z = mfi.nodaltilebox(2););

        amrex::ParallelFor(nodal_x, nodal_y, nodal_z,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            facex(i,j,k) = 0.5*(cc(i,j,k)+cc(i-1,j,k));
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            facey(i,j,k) = 0.5*(cc(i,j,k)+cc(i,j-1,k));
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            facez(i,j,k) = 0.5*(cc(i,j,k)+cc(i,j,k-1));
        });
    }
#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************Multifab_Manipulation::AverageCellCenteredMultiFabToCellFaces()************************\n";
#endif

}

void AverageFaceCenteredMultiFabToCellCenters(std::array< amrex::MultiFab,
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
            cc(i,j,k,0) = 0.5*(facex(i,j,k)+facex(i+1,j,k));
            cc(i,j,k,1) = 0.5*(facey(i,j,k)+facey(i+1,j,k));
            cc(i,j,k,2) = 0.5*(facez(i,j,k)+facez(i+1,j,k));
        });
    }

}

#ifdef AMREX_USE_EB
void
Multifab_Manipulation::SpecifyValueOnlyOnCutcells(amrex::MultiFab& mf, amrex::Real const value) 
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::SpecifyValueOnlyOnCutcells************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    auto factory  = dynamic_cast<amrex::EBFArrayBoxFactory const*>(&(mf.Factory()));

    auto const &flags = factory->getMultiEBCellFlagFab();
    auto const &vfrac = factory->getVolFrac();

    auto iv = mf.ixType().toIntVect();

    for ( amrex::MFIter mfi(flags, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) 
    {
        const auto& box = mfi.tilebox( iv, mf.nGrowVect() ); 

        auto const& mf_array =  mf.array(mfi); 

        amrex::FabType fab_type = flags[mfi].getType(box);

        if(fab_type == amrex::FabType::regular) 
        {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
               mf_array(i, j, k) = amrex::Real(0.);
            });
        }
        else if (fab_type == amrex::FabType::covered) 
        {
            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
               mf_array(i, j, k) = amrex::Real(0.);
            });
        }
        else //box contains some cutcells
        {
            auto const &vfrac_array = vfrac.const_array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
               if(vfrac_array(i,j,k) > 0 and vfrac_array(i,j,k) < 1) 
               {
                   mf_array(i, j, k) = value;
               } 
            });
        }
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************Multifab_Manipulation::SpecifyValueOnlyOnCutcells************************\n";
#endif
}


void
Multifab_Manipulation::CopyValuesIntoAMultiFabOnCutcells(amrex::MultiFab& target_mf, amrex::MultiFab& source_mf) 
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::CopyValuesIntoAMultiFabOnCutcells************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif
    /*target_mf is initialized to 0 and contains cutcell information through its factory*/ 
    /*source_mf is a regular mf with some field information that we would like to copy into target_mf*/ 

    auto factory  = dynamic_cast<amrex::EBFArrayBoxFactory const*>(&(target_mf.Factory()));

    auto const &flags = factory->getMultiEBCellFlagFab();
    auto const &vfrac = factory->getVolFrac();

    auto iv = target_mf.ixType().toIntVect();

    for ( amrex::MFIter mfi(flags, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) 
    {
        const auto& box = mfi.tilebox( iv, target_mf.nGrowVect() ); 

        auto const& target_mf_array =  target_mf.array(mfi); 
        auto const& source_mf_array =  source_mf.array(mfi); 

        amrex::FabType fab_type = flags[mfi].getType(box);
        if((fab_type != amrex::FabType::regular) && (fab_type != amrex::FabType::covered))
        {
            /*box has some cutcells*/
            auto const &vfrac_array = vfrac.const_array(mfi);

            amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
               if(vfrac_array(i,j,k) > 0.+VFRAC_THREASHOLD and vfrac_array(i,j,k) < 1.-VFRAC_THREASHOLD) 
               {
                   target_mf_array(i, j, k) = source_mf_array(i,j,k);
               } 
            });
        }
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************Multifab_Manipulation::CopyValuesIntoAMultiFabOnCutcells************************\n";
#endif
}


amrex::Real
Multifab_Manipulation::GetTotalNumberOfCutcells(amrex::MultiFab& mf) 
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************Multifab_Manipulation::GetTotalNumberOfCutcells************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif
    /*mf must be created with a factory that holds cutcell info*/

    auto factory  = dynamic_cast<amrex::EBFArrayBoxFactory const*>(&(mf.Factory()));

    auto const &flags = factory->getMultiEBCellFlagFab();
    auto const &vfrac = factory->getVolFrac();

    ReduceOps<ReduceOpSum> reduce_op;
    ReduceData<Real> reduce_data(reduce_op);
    using ReduceTuple = typename decltype(reduce_data)::Type;

    for ( amrex::MFIter mfi(flags, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi ) 
    {
        const auto& box = mfi.tilebox(); 

        auto const& mf_array =  mf.array(mfi); 

        amrex::FabType fab_type = flags[mfi].getType(box);
        if((fab_type != amrex::FabType::regular) && (fab_type != amrex::FabType::covered))
        {
            /*box has some cutcells*/
            auto const &vfrac_array = vfrac.const_array(mfi);

            reduce_op.eval(box, reduce_data,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple
            {
               amrex::Real weight = (vfrac_array(i,j,k) >= 0.+VFRAC_THREASHOLD and vfrac_array(i,j,k) <= 1.0-VFRAC_THREASHOLD) ? 1.0 : 0;
               return weight;   
            });
        }
    }
    amrex::Real sum = amrex::get<0>(reduce_data.value());
    ParallelDescriptor::ReduceRealSum(sum);

    return sum;
#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************Multifab_Manipulation::GetTotalNumberOfCutcells************************\n";
#endif
}


#endif //for AMREX_USE_EB

amrex::RealArray vecToArr(amrex::Vector<amrex::Real>& vec)
{
    amrex::RealArray array;
    for (int i=0; i<AMREX_SPACEDIM; ++i) array[i] = vec[i];
    return array;
}

