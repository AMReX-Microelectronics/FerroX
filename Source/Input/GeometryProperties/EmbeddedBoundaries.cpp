/* Contributors: Saurabh Sawant, Prabhat Kumar
 *
 */
#include "GeometryProperties.H"

#include "../../Utils/SelectWarpXUtils/WarpXUtil.H"
#include "../../Utils/FerroXUtils/FerroXUtil.H"

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>
#include <AMReX_RealBox.H>
#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>
#include <AMReX_EBSupport.H>
#include <AMReX_EB2_IndexSpace_STL.H>


using namespace amrex;

//template<typename T>
//class TD;

c_EmbeddedBoundaries::c_EmbeddedBoundaries()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_EmbeddedBoundaries() Constructor************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    ReadGeometry();

#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t}************************c_EmbeddedBoundaries() Constructor************************\n";
#endif
}


c_EmbeddedBoundaries::~c_EmbeddedBoundaries()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t{************************c_EmbeddedBoundaries() Destructor************************\n";
    amrex::Print() << "\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    m_p_soln_mf.clear();
    //m_p_beta_mf.clear();
    m_p_factory.clear();
    vec_object_names.clear();
    map_basic_objects_type.clear();
    map_basic_objects_info.clear();
    map_basic_objects_soln.clear();
    map_basic_objects_beta.clear(); 

    geom = nullptr;
    ba = nullptr;
    dm = nullptr;

#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t}************************c_EmbeddedBoundaries() Destructor************************\n";
#endif
}


void
c_EmbeddedBoundaries::ReadGeometry()
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************c_EmbeddedBoundaries()::ReadGeometry()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
    std::string prt = "\t\t\t\t\t";
#endif

    //setting default values
    required_coarsening_level = 0; // max amr level (at present 0)
    max_coarsening_level = 100;    // typically a huge number so MG coarsens as much as possible
    support = EBSupport::full;
    std::string eb_support_str = "full";
    specify_input_using_eb2 = 0;
    specify_separate_surf_soln = 1;
    specify_inhomogeneous_dirichlet = 0;

    amrex::ParmParse pp_ebgeom("ebgeom");
    pp_ebgeom.query("required_coarsening_level", required_coarsening_level);
    pp_ebgeom.query("max_coarsening_level", max_coarsening_level);
    pp_ebgeom.query("support", eb_support_str);
    support = map_eb_support[eb_support_str];
    pp_ebgeom.query("specify_input_using_eb2", specify_input_using_eb2);
    pp_ebgeom.query("specify_inhomo_dir", specify_inhomogeneous_dirichlet);

    if(specify_input_using_eb2 == 1) {
       specify_separate_surf_soln = 0; 
    }

    amrex::Print() << "\n##### EMBEDDED BOUNDARY PROPERTIES #####\n\n";
    amrex::Print() << "##### ebgeom.required_coarsening_level: " << required_coarsening_level << "\n";
    amrex::Print() << "##### ebgeom.max_coarsening_level: " << max_coarsening_level << "\n";
    amrex::Print() << "##### ebgeom.support: " << eb_support_str << "\n";
    amrex::Print() << "##### ebgeom.specify_input_using_eb2: " << specify_input_using_eb2 << "\n";
    amrex::Print() << "##### ebgeom.specify_inhomo_dir: " << specify_inhomogeneous_dirichlet << "\n";
    amrex::Print() << "##### ebgeom.specify_separate_surf_soln: " << specify_separate_surf_soln << "\n";
    if(specify_inhomogeneous_dirichlet == 1 && specify_separate_surf_soln == 0) 
    {
       getWithParser(pp_ebgeom,"surf_soln", surf_soln);
       amrex::Print() << "##### ebgeom.surf_soln: " << surf_soln << "\n";
    }

    if(!specify_input_using_eb2) 
    {
        num_objects = 0;
        bool basic_objects_specified = pp_ebgeom.queryarr("objects", vec_object_names);
        int c=0;
        for (auto it: vec_object_names)
        {
            if (map_basic_objects_type.find(it) == map_basic_objects_type.end()) {
                amrex::ParmParse pp_object(it);

                pp_object.get("geom_type", map_basic_objects_type[it]);
                ReadObjectInfo(it, map_basic_objects_type[it], pp_object);

                if(specify_inhomogeneous_dirichlet == 1) 
                {
                    getWithParser(pp_object,"surf_soln", map_basic_objects_soln[it]);
                    amrex::Print()  << "##### surf_soln: " << map_basic_objects_soln[it] << "\n";
                } 

                ++c;
            }
        }
        num_objects = c;
        amrex::Print() << "\n##### total number of basic objects: " << num_objects << "\n";
    }

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************c_EmbeddedBoundaries()::ReadGeometry()************************\n";
#endif
}


void
c_EmbeddedBoundaries::ReadObjectInfo(std::string object_name, std::string object_type, amrex::ParmParse pp_object)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t\t{************************c_EmbeddedBoundaries()::ReadObjectInfo()************************\n";
    amrex::Print() << "\t\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    amrex::Print() << "\n##### Object name: " << object_name << "\n";
    amrex::Print() << "##### Object type: " << object_type << "\n";

    switch (map_object_type_enum[object_type]) 
    {  
        case s_ObjectType::object::box:
        {
            amrex::Vector<amrex::Real> lo;
            getArrWithParser(pp_object,"box_lo", lo,0,AMREX_SPACEDIM);
        
            amrex::Print() << "##### box_lo: ";
            for (int i=0; i<AMREX_SPACEDIM; ++i) amrex::Print() << lo[i] << "  ";
            amrex::Print() << "\n";


            amrex::Vector<amrex::Real> hi;
            getArrWithParser(pp_object,"box_hi", hi,0,AMREX_SPACEDIM);
        
            amrex::Print() << "##### box_hi: ";
            for (int i=0; i<AMREX_SPACEDIM; ++i) amrex::Print() << hi[i] << "  ";
            amrex::Print() << "\n";


            bool has_fluid_inside;
            pp_object.get("has_fluid_inside", has_fluid_inside);

            amrex::Print() << "##### Box has_fluid_inside?: " << has_fluid_inside << "\n";


            amrex::EB2::BoxIF box(vecToArr(lo), vecToArr(hi), has_fluid_inside);

            map_basic_objects_info[object_name] = box;  
            break;
        }
        case s_ObjectType::object::plane:
        {
            amrex::Vector<amrex::Real> point;
            getArrWithParser(pp_object, "point", point, 0, AMREX_SPACEDIM);

            amrex::Print() << "##### plane point: ";
            for (int i=0; i<AMREX_SPACEDIM; ++i) amrex::Print() << point[i] << "  ";
            amrex::Print() << "\n";

            amrex::Vector<amrex::Real> normal;
            getArrWithParser(pp_object, "normal", normal, 0, AMREX_SPACEDIM);

            amrex::Print() << "##### plane normal: ";
            for (int i=0; i<AMREX_SPACEDIM; ++i) amrex::Print() << normal[i] << "  ";
            amrex::Print() << "\n";

            amrex::EB2::PlaneIF plane(vecToArr(point), vecToArr(normal));

            map_basic_objects_info[object_name] = plane;  
            break;
        }
        case s_ObjectType::object::parser:
        {
            s_PARSER object;

            pp_object.get("parser_function", object.parser_function_str);
            amrex::Print() << "##### parser_function: " << object.parser_function_str << "\n";

            map_basic_objects_info[object_name] = object;  
            break;
        }
        case s_ObjectType::object::stl:
        {
            amrex::Abort("No support for geom_type " + object_type + " at present. Use eb2 to define the stl file.");

            s_STL stl;

            pp_object.get("file", stl.file);
            amrex::Print() << "##### stl file: " << stl.file << "\n";

            pp_object.queryAdd("scale", stl.scale);
            amrex::Print() << "##### stl scale: " << stl.scale << "\n";

            pp_object.queryAdd("center", stl.center);
            amrex::Print() << "##### stl center: ";
            for (int i=0; i<AMREX_SPACEDIM; ++i) amrex::Print() << stl.center[i] << "  ";
            amrex::Print() << "\n";

            pp_object.queryAdd("reverse_normal", stl.reverse_normal);
            amrex::Print() << "##### stl reverse_normal: " << stl.reverse_normal << "\n";

            map_basic_objects_info[object_name] = stl;  
            break;
        }
        default:
        {
            amrex::Abort("geom_type " + object_type + " not supported");
            break;
        }
    }
#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t\t}************************c_EmbeddedBoundaries()::ReadObjectInfo()************************\n";
#endif
}


void
c_EmbeddedBoundaries::BuildGeometry(const amrex::Geometry* GEOM, const amrex::BoxArray* BA, const amrex::DistributionMapping* DM)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t{************************c_EmbeddedBoundaries()::BuildObjects()************************\n";
    amrex::Print() << "\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
    std::string prt = "\t\t\t\t";
#endif

    geom = GEOM;
    ba = BA;
    dm = DM;

    if(specify_input_using_eb2)
    {
        amrex::EB2::Build(*geom, required_coarsening_level, max_coarsening_level);

        const auto& eb_is = EB2::IndexSpace::top();
        const auto& eb_level = eb_is.getLevel(*geom);
        Vector<int> ng_ebs = {2,2,2};
        p_factory_union = amrex::makeEBFabFactory(&eb_level, *ba, *dm, ng_ebs, support);

        if(specify_inhomogeneous_dirichlet == 1) 
        {
            p_surf_soln_union = std::make_unique<amrex::MultiFab>(*ba, *dm, 1, 0, MFInfo(), *p_factory_union); 
            p_surf_soln_union->setVal(surf_soln);
        }
    }
    else 
    {
        Vector<int> ng_ebs = {2,2,2};
        
        if(specify_inhomogeneous_dirichlet == 1)  m_p_soln_mf.resize(num_objects);

        int c=0;
        for (auto it: map_basic_objects_type)
        {
            std::string name = it.first; 
            std::string geom_type = it.second;
              
            #ifdef PRINT_LOW
            amrex::Print() << prt << "\nname: " << name << ", geom_type: " << geom_type << "\n";
            #endif 

            switch (map_object_type_enum[geom_type]) 
            {  
                case s_ObjectType::object::box:
                {
                    using ObjectType = amrex::EB2::BoxIF;
                    BuildSingleObject<ObjectType>(name);
                    break;
                }
                case s_ObjectType::object::plane:
                {
                    using ObjectType = amrex::EB2::PlaneIF;
                    BuildSingleObject<ObjectType>(name);
                    break;
                }
                case s_ObjectType::object::parser:
                {
                    using ObjectType = s_PARSER;
                    BuildSingleParserObject<ObjectType>(name);
                    break;
                }
                case s_ObjectType::object::stl:
                {
                    using ObjectType = s_STL;
                    BuildSingleSTLObject<ObjectType>(name);
                    break;
                }
            }

            if(specify_inhomogeneous_dirichlet == 1) 
            {
                m_p_soln_mf[c] = std::make_unique<amrex::MultiFab>(*ba, *dm, 1, 0, MFInfo(), *m_p_factory[c]); 
                (*m_p_soln_mf[c]).setVal(0.); 
                Multifab_Manipulation::SpecifyValueOnlyOnCutcells(*m_p_soln_mf[c], map_basic_objects_soln[name]);

                #ifdef PRINT_LOW
                amrex::Print() << prt << "Index space size : " << amrex::EB2::IndexSpace::size() << "\n";
                #endif

                if(num_objects > 1) amrex::EB2::IndexSpace::clear();
            }
            ++c;
        }

        if(num_objects == 1) 
        {
            p_factory_union = std::move(m_p_factory[0]);
        }
        else if(num_objects == 2 || num_objects == 3) 
        {  
            auto name1 = vec_object_names[0];  
            auto geom_type1 = map_basic_objects_type[name1];  
            auto name2 = vec_object_names[1];
            auto geom_type2 = map_basic_objects_type[name2];  
            auto name3 = vec_object_names[2];
            auto geom_type3 = map_basic_objects_type[name3];  

	    if(num_objects == 2)
	    {
                if ( (map_object_type_enum[geom_type1] == s_ObjectType::object::box) &&
                     (map_object_type_enum[geom_type2] == s_ObjectType::object::box) )
                {
                    using ObjectType1 = amrex::EB2::BoxIF;
                    using ObjectType2 = amrex::EB2::BoxIF;

                    BuildUnionObject<ObjectType1, ObjectType2>(name1, name2);
                }
                else if ( (map_object_type_enum[geom_type1] == s_ObjectType::object::parser) &&
                          (map_object_type_enum[geom_type2] == s_ObjectType::object::parser) )
                {
                    using ObjectType1 = s_PARSER;
                    using ObjectType2 = s_PARSER;

                    BuildUnionParserObject<ObjectType1, ObjectType2>(name1, name2);
                }
	    } 
	    if(num_objects == 3)
	    {
                if ( (map_object_type_enum[geom_type1] == s_ObjectType::object::box) &&
                          (map_object_type_enum[geom_type2] == s_ObjectType::object::box) &&
                          (map_object_type_enum[geom_type3] == s_ObjectType::object::box) )
                {
                     using ObjectType1 = amrex::EB2::BoxIF;
                     using ObjectType2 = amrex::EB2::BoxIF;
                     using ObjectType3 = amrex::EB2::BoxIF;

                     BuildUnionOfUnionObject<ObjectType1, ObjectType2, ObjectType3>(name1, name2, name3);
                }
	    }
	}

        if(specify_inhomogeneous_dirichlet == 1)
        {
            p_surf_soln_union = std::make_unique<amrex::MultiFab>(*ba, *dm, 1, 0, MFInfo(), *p_factory_union);

            p_surf_soln_union->setVal(0);    
            for(int i=0; i < num_objects; ++i)
            {
                p_surf_soln_union->plus(get_soln_mf(i), 0, 1, 0);
            }
            clear_soln_mf();
        }
    }
#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t}************************c_EmbeddedBoundaries()::BuildObjects()************************\n";
#endif
}


template<typename ObjectType>
void
c_EmbeddedBoundaries::BuildSingleObject(std::string name)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************c_EmbeddedBoundaries()::BuildSingleObject()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    auto object = std::any_cast<ObjectType>(map_basic_objects_info[name]);
    auto gshop = amrex::EB2::makeShop(object);
    amrex::EB2::Build(gshop, *geom, required_coarsening_level, max_coarsening_level);
    
    const auto& eb_is = EB2::IndexSpace::top();
    const auto& eb_level = eb_is.getLevel(*geom);
    Vector<int> ng_ebs = {2,2,2};

    m_p_factory.push_back(amrex::makeEBFabFactory(&eb_level, *ba, *dm, ng_ebs, support));

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************c_EmbeddedBoundaries()::BuildSingleObject()************************\n";
#endif
}


template<typename ObjectType>
void
c_EmbeddedBoundaries::BuildSingleParserObject(std::string name)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************c_EmbeddedBoundaries()::BuildSingleParserObject()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    auto object = std::any_cast<ObjectType>(map_basic_objects_info[name]);

    amrex::Parser parser = makeParser(object.parser_function_str, {"x", "y", "z"});
    amrex::EB2::ParserIF pif(parser.compile<3>());
    auto gshop = amrex::EB2::makeShop(pif, parser);
    amrex::EB2::Build(gshop, *geom, required_coarsening_level, max_coarsening_level);

    const auto& eb_is = EB2::IndexSpace::top();
    const auto& eb_level = eb_is.getLevel(*geom);
    Vector<int> ng_ebs = {2,2,2};

    m_p_factory.push_back(amrex::makeEBFabFactory(&eb_level, *ba, *dm, ng_ebs, support));

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************c_EmbeddedBoundaries()::BuildSingleParserObject()************************\n";
#endif
}


template<typename ObjectType>
void
c_EmbeddedBoundaries::BuildSingleSTLObject(std::string name)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************c_EmbeddedBoundaries()::BuildSingleSTLObject()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    Vector<int> ng_ebs = {2,2,2};

    const auto& eb_is = EB2::IndexSpace::top();
    const auto& eb_level = eb_is.getLevel(*geom);

    m_p_factory.push_back(amrex::makeEBFabFactory(&eb_level, *ba, *dm, ng_ebs, support));

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************c_EmbeddedBoundaries()::BuildSingleSTLObject()************************\n";
#endif
}


template<typename ObjectType1, typename ObjectType2>
void
c_EmbeddedBoundaries::BuildUnionObject(std::string name1, std::string name2)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************c_EmbeddedBoundaries()::BuildUnionObject()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    auto object1 = std::any_cast<ObjectType1>(map_basic_objects_info[name1]);
    auto object2 = std::any_cast<ObjectType2>(map_basic_objects_info[name2]);
    auto union_object = amrex::EB2::makeUnion(object1, object2);

    auto gshop = amrex::EB2::makeShop(union_object);
    amrex::EB2::Build(gshop, *geom, required_coarsening_level, max_coarsening_level);
    const auto& eb_is = EB2::IndexSpace::top();
    const auto& eb_level = eb_is.getLevel(*geom);
    Vector<int> ng_ebs = {2,2,2};

    p_factory_union = amrex::makeEBFabFactory(&eb_level, *ba, *dm, ng_ebs, support);

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************c_EmbeddedBoundaries()::BuildUnionObject()************************\n";
#endif
}

template<typename ObjectType1, typename ObjectType2, typename ObjectType3>
void
c_EmbeddedBoundaries::BuildUnionOfUnionObject(std::string name1, std::string name2, std::string name3)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************c_EmbeddedBoundaries()::BuildUnionOfUnionObject()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    auto object1 = std::any_cast<ObjectType1>(map_basic_objects_info[name1]);
    auto object2 = std::any_cast<ObjectType2>(map_basic_objects_info[name2]);
    auto object3 = std::any_cast<ObjectType3>(map_basic_objects_info[name3]);
    auto union_object1 = amrex::EB2::makeUnion(object1, object2);
    auto union_object = amrex::EB2::makeUnion(union_object1, object3);

    auto gshop = amrex::EB2::makeShop(union_object);
    amrex::EB2::Build(gshop, *geom, required_coarsening_level, max_coarsening_level);
    const auto& eb_is = EB2::IndexSpace::top();
    const auto& eb_level = eb_is.getLevel(*geom);
    Vector<int> ng_ebs = {2,2,2};

    p_factory_union = amrex::makeEBFabFactory(&eb_level, *ba, *dm, ng_ebs, support);

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************c_EmbeddedBoundaries()::BuildUnioniOfUnionObject()************************\n";
#endif
}


template<typename ObjectType1, typename ObjectType2>
void
c_EmbeddedBoundaries::BuildUnionParserObject(std::string name1, std::string name2)
{
#ifdef PRINT_NAME
    amrex::Print() << "\n\n\t\t\t\t\t{************************c_EmbeddedBoundaries()::BuildUnionObject()************************\n";
    amrex::Print() << "\t\t\t\t\tin file: " << __FILE__ << " at line: " << __LINE__ << "\n";
#endif

    auto object1 = std::any_cast<ObjectType1>(map_basic_objects_info[name1]);
    amrex::Parser parser1 = makeParser(object1.parser_function_str, {"x", "y", "z"});
    amrex::EB2::ParserIF pif1(parser1.compile<3>());
    
    auto object2 = std::any_cast<ObjectType2>(map_basic_objects_info[name2]);
    amrex::Parser parser2 = makeParser(object2.parser_function_str, {"x", "y", "z"});
    amrex::EB2::ParserIF pif2(parser2.compile<3>());
 
    //amrex::Parser union_parser = parser1 + parser2;
    auto union_object = amrex::EB2::makeUnion(pif1, pif2);

    auto gshop = amrex::EB2::makeShop(union_object);
    amrex::EB2::Build(gshop, *geom, required_coarsening_level, max_coarsening_level);
    const auto& eb_is = EB2::IndexSpace::top();
    const auto& eb_level = eb_is.getLevel(*geom);
    Vector<int> ng_ebs = {2,2,2};

    p_factory_union = amrex::makeEBFabFactory(&eb_level, *ba, *dm, ng_ebs, support);

#ifdef PRINT_NAME
    amrex::Print() << "\t\t\t\t\t}************************c_EmbeddedBoundaries()::BuildUnionObject()************************\n";
#endif
}
