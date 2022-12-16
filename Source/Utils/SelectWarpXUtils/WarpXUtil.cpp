/* Copyright 2019-2020 Andrew Myers, Burlen Loring, Luca Fedeli
 * Maxence Thevenet, Remi Lehe, Revathi Jambunathan
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

//#include "TextMsg.H"
#include "WarpXConst.H"
#include "WarpXUtil.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLassert.H>
#include <AMReX_Box.H>
#include <AMReX_Config.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_FabArray.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Parser.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <limits>

using namespace amrex;

void PreparseAMReXInputIntArray(amrex::ParmParse& a_pp, char const * const input_str, const bool replace)
{
    const int cnt = a_pp.countval(input_str);
    if (cnt > 0) {
        Vector<int> input_array;
        getArrWithParser(a_pp, input_str, input_array);
        if (replace) {
            a_pp.remove(input_str);
        }
        a_pp.addarr(input_str, input_array);
    }
}

void ParseGeometryInput()
{
    // Ensure that geometry.dims is set properly.

    // Parse prob_lo and hi, evaluating any expressions since geometry does not
    // parse its input
    ParmParse pp_geometry("geometry");

    Vector<Real> prob_lo(AMREX_SPACEDIM);
    Vector<Real> prob_hi(AMREX_SPACEDIM);

    getArrWithParser(pp_geometry, "prob_lo", prob_lo, 0, AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(prob_lo.size() == AMREX_SPACEDIM);
    getArrWithParser(pp_geometry, "prob_hi", prob_hi, 0, AMREX_SPACEDIM);
    AMREX_ALWAYS_ASSERT(prob_hi.size() == AMREX_SPACEDIM);

    pp_geometry.addarr("prob_lo", prob_lo);
    pp_geometry.addarr("prob_hi", prob_hi);

    // Parse amr input, evaluating any expressions since amr does not parse its input
    ParmParse pp_amr("amr");

    // Note that n_cell is replaced so that only the parsed version is written out to the
    // warpx_job_info file. This must be done since yt expects to be able to parse
    // the value of n_cell from that file. For the rest, this doesn't matter.
    PreparseAMReXInputIntArray(pp_amr, "n_cell", true);
    PreparseAMReXInputIntArray(pp_amr, "max_grid_size", false);
    PreparseAMReXInputIntArray(pp_amr, "max_grid_size_x", false);
    PreparseAMReXInputIntArray(pp_amr, "max_grid_size_y", false);
    PreparseAMReXInputIntArray(pp_amr, "max_grid_size_z", false);
    PreparseAMReXInputIntArray(pp_amr, "blocking_factor", false);
    PreparseAMReXInputIntArray(pp_amr, "blocking_factor_x", false);
    PreparseAMReXInputIntArray(pp_amr, "blocking_factor_y", false);
    PreparseAMReXInputIntArray(pp_amr, "blocking_factor_z", false);
}

void Store_parserString(const amrex::ParmParse& pp, std::string query_string,
                        std::string& stored_string)
{
    std::vector<std::string> f;
    pp.getarr(query_string.c_str(), f);
    stored_string.clear();
    for (auto const& s : f) {
        stored_string += s;
    }
    f.clear();
}


Parser makeParser (std::string const& parse_function, amrex::Vector<std::string> const& varnames)
{
    // Since queryWithParser recursively calls this routine, keep track of symbols
    // in case an infinite recursion is found (a symbol's value depending on itself).
    static std::set<std::string> recursive_symbols;

    Parser parser(parse_function);
    parser.registerVariables(varnames);

    std::set<std::string> symbols = parser.symbols();
    for (auto const& v : varnames) symbols.erase(v.c_str());

    // User can provide inputs under this name, through which expressions
    // can be provided for arbitrary variables. PICMI inputs are aware of
    // this convention and use the same prefix as well. This potentially
    // includes variable names that match physical or mathematical
    // constants, in case the user wishes to enforce a different
    // system of units or some form of quasi-physical behavior in the
    // simulation. Thus, this needs to override any built-in
    // constants.
    ParmParse pp_my_constants("my_constants");

    // Physical / Numerical Constants available to parsed expressions
    static std::map<std::string, amrex::Real> warpx_constants =
      {
       {"clight", PhysConst::c},
       {"epsilon0", PhysConst::ep0},
       {"mu0", PhysConst::mu0},
       {"q_e", PhysConst::q_e},
       {"m_e", PhysConst::m_e},
       {"m_p", PhysConst::m_p},
       {"m_u", PhysConst::m_u},
       {"kb", PhysConst::kb},
       {"pi", MathConst::pi},
      };

    for (auto it = symbols.begin(); it != symbols.end(); ) {
        // Always parsing in double precision avoids potential overflows that may occur when parsing
        // user's expressions because of the limited range of exponentials in single precision
        double v;

        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            recursive_symbols.count(*it)==0,
            "Expressions contains recursive symbol "+*it);
        recursive_symbols.insert(*it);
        const bool is_input = queryWithParser(pp_my_constants, it->c_str(), v);
        recursive_symbols.erase(*it);

        if (is_input) {
            parser.setConstant(*it, v);
            it = symbols.erase(it);
            continue;
        }

        auto constant = warpx_constants.find(*it);
        if (constant != warpx_constants.end()) {
          parser.setConstant(*it, constant->second);
          it = symbols.erase(it);
          continue;
        }

        ++it;
    }
//    for (auto const& s : symbols) {
//        amrex::Abort(Utils::TextMsg::Err("makeParser::Unknown symbol "+s));
//    }
    return parser;
}

double
parseStringtoReal(std::string str)
{
    auto parser = makeParser(str, {});
    auto exe = parser.compileHost<0>();
    double result = exe();
    return result;
}

int
parseStringtoInt(std::string str, std::string name)
{
    auto const rval = static_cast<amrex::Real>(parseStringtoReal(str));
    int ival = safeCastToInt(std::round(rval), name);
    return ival;
}

// Overloads for float/double instead of amrex::Real to allow makeParser() to query for
// my_constants as double even in single precision mode
// Always parsing in double precision avoids potential overflows that may occur when parsing user's
// expressions because of the limited range of exponentials in single precision
int
queryWithParser (const amrex::ParmParse& a_pp, char const * const str, float& val)
{
    // call amrex::ParmParse::query, check if the user specified str.
    std::string tmp_str;
    int is_specified = a_pp.query(str, tmp_str);
    if (is_specified)
    {
        // If so, create a parser object and apply it to the value provided by the user.
        std::string str_val;
        Store_parserString(a_pp, str, str_val);
        val = static_cast<float>(parseStringtoReal(str_val));
    }
    // return the same output as amrex::ParmParse::query
    return is_specified;
}

void
getWithParser (const amrex::ParmParse& a_pp, char const * const str, float& val)
{
    // If so, create a parser object and apply it to the value provided by the user.
    std::string str_val;
    Store_parserString(a_pp, str, str_val);
    val = static_cast<float>(parseStringtoReal(str_val));
}

int
queryWithParser (const amrex::ParmParse& a_pp, char const * const str, double& val)
{
    // call amrex::ParmParse::query, check if the user specified str.
    std::string tmp_str;
    int is_specified = a_pp.query(str, tmp_str);
    if (is_specified)
    {
        // If so, create a parser object and apply it to the value provided by the user.
        std::string str_val;
        Store_parserString(a_pp, str, str_val);
        val = parseStringtoReal(str_val);
    }
    // return the same output as amrex::ParmParse::query
    return is_specified;
}

void
getWithParser (const amrex::ParmParse& a_pp, char const * const str, double& val)
{
    // If so, create a parser object and apply it to the value provided by the user.
    std::string str_val;
    Store_parserString(a_pp, str, str_val);
    val = parseStringtoReal(str_val);
}

int
queryArrWithParser (const amrex::ParmParse& a_pp, char const * const str, std::vector<amrex::Real>& val,
                    const int start_ix, const int num_val)
{
    // call amrex::ParmParse::query, check if the user specified str.
    std::vector<std::string> tmp_str_arr;
    int is_specified = a_pp.queryarr(str, tmp_str_arr, start_ix, num_val);
    if (is_specified)
    {
        // If so, create parser objects and apply them to the values provided by the user.
        int const n = static_cast<int>(tmp_str_arr.size());
        val.resize(n);
        for (int i=0 ; i < n ; i++) {
            val[i] = static_cast<amrex::Real>(parseStringtoReal(tmp_str_arr[i]));
        }
    }
    // return the same output as amrex::ParmParse::query
    return is_specified;
}

void
getArrWithParser (const amrex::ParmParse& a_pp, char const * const str, std::vector<amrex::Real>& val,
                    const int start_ix, const int num_val)
{
    // Create parser objects and apply them to the values provided by the user.
    std::vector<std::string> tmp_str_arr;
    a_pp.getarr(str, tmp_str_arr, start_ix, num_val);

    int const n = static_cast<int>(tmp_str_arr.size());
    val.resize(n);
    for (int i=0 ; i < n ; i++) {
        val[i] = static_cast<amrex::Real>(parseStringtoReal(tmp_str_arr[i]));
    }
}

int queryWithParser (const amrex::ParmParse& a_pp, char const * const str, int& val) {
    amrex::Real rval;
    const int result = queryWithParser(a_pp, str, rval);
    if (result) {
        val = safeCastToInt(std::round(rval), str);
    }
    return result;
}

void getWithParser (const amrex::ParmParse& a_pp, char const * const str, int& val) {
    amrex::Real rval;
    getWithParser(a_pp, str, rval);
    val = safeCastToInt(std::round(rval), str);
}

int queryArrWithParser (const amrex::ParmParse& a_pp, char const * const str, std::vector<int>& val,
                        const int start_ix, const int num_val) {
    std::vector<amrex::Real> rval;
    const int result = queryArrWithParser(a_pp, str, rval, start_ix, num_val);
    if (result) {
        val.resize(rval.size());
        for (unsigned long i = 0 ; i < val.size() ; i++) {
            val[i] = safeCastToInt(std::round(rval[i]), str);
        }
    }
    return result;
}

void getArrWithParser (const amrex::ParmParse& a_pp, char const * const str, std::vector<int>& val,
                       const int start_ix, const int num_val) {
    std::vector<amrex::Real> rval;
    getArrWithParser(a_pp, str, rval, start_ix, num_val);
    val.resize(rval.size());
    for (unsigned long i = 0 ; i < val.size() ; i++) {
        val[i] = safeCastToInt(std::round(rval[i]), str);
    }
}

int safeCastToInt(const amrex::Real x, const std::string& real_name) {
    int result = 0;
    bool error_detected = false;
    std::string assert_msg;
    // (2.0*(numeric_limits<int>::max()/2+1)) converts numeric_limits<int>::max()+1 to a real ensuring accuracy to all digits
    // This accepts x = 2**31-1 but rejects 2**31.
    using namespace amrex::literals;
    constexpr amrex::Real max_range = (2.0_rt*static_cast<amrex::Real>(std::numeric_limits<int>::max()/2+1));
    if (x < max_range) {
        if (std::ceil(x) >= std::numeric_limits<int>::min()) {
            result = static_cast<int>(x);
        } else {
            error_detected = true;
            assert_msg = "Negative overflow detected when casting " + real_name + " = " + std::to_string(x) + " to int";
        }
    } else if (x > 0) {
        error_detected = true;
        assert_msg =  "Overflow detected when casting " + real_name + " = " + std::to_string(x) + " to int";
    } else {
        error_detected = true;
        assert_msg =  "NaN detected when casting " + real_name + " to int";
    }
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(!error_detected, assert_msg);
    return result;
}

