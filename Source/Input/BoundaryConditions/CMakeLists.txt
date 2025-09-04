foreach(D IN LISTS FerroX_DIMS)
    ferrox_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        BoundaryConditions.cpp
    )
endforeach()
