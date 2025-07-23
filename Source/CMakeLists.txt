foreach(D IN LISTS FerroX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        main.cpp
        FerroX.cpp
        Plotfile.cpp
    )
endforeach()

