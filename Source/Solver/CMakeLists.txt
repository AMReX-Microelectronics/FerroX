foreach(D IN LISTS FerroX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        ElectrostaticSolver.cpp
        Initialization.cpp
        ChargeDensity.cpp
        TotalEnergyDensity.cpp
    )
endforeach()
