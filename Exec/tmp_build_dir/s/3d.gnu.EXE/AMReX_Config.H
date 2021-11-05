#ifndef AMREX_HAVE_NO_CONFIG_H
#define AMREX_HAVE_NO_CONFIG_H
#ifndef AMREX_GIT_VERSION
#define AMREX_GIT_VERSION "21.08-36-g2788e5581562"
#endif
#ifndef BL_SPACEDIM
#define BL_SPACEDIM 3
#endif
#ifndef AMREX_SPACEDIM
#define AMREX_SPACEDIM 3
#endif
#ifndef BL_FORT_USE_UNDERSCORE
#define BL_FORT_USE_UNDERSCORE 1
#endif
#ifndef AMREX_FORT_USE_UNDERSCORE
#define AMREX_FORT_USE_UNDERSCORE 1
#endif
#ifndef BL_Linux
#define BL_Linux 1
#endif
#ifndef AMREX_Linux
#define AMREX_Linux 1
#endif
#ifndef NDEBUG
#define NDEBUG 1
#endif
#ifdef __cplusplus
#endif
#if defined(AMREX_USE_OMP) && !defined(_OPENMP)
#error libamrex was built with OpenMP
#endif
#endif
