














module amrex_lo_bctypes_module

  implicit none
  integer, parameter :: amrex_lo_dirichlet         = 101
  integer, parameter :: amrex_lo_neumann           = 102
  integer, parameter :: amrex_lo_reflect_odd       = 103
  integer, parameter :: amrex_lo_marshak           = 104
  integer, parameter :: amrex_lo_sanchez_pomraning = 105
  integer, parameter :: amrex_lo_inflow            = 106
  integer, parameter :: amrex_lo_periodic          = 200
  integer, parameter :: amrex_lo_bogus             = 1729

end module amrex_lo_bctypes_module
