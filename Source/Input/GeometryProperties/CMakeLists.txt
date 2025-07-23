foreach(D IN LISTS FerroX_DIMS)
    warpx_set_suffix_dims(SD ${D})
    target_sources(lib_${SD}
      PRIVATE
        GeometryProperties.cpp
    )
    
    if(FerroX_EB)
        target_sources(lib_${SD}
          PRIVATE
            EmbeddedBoundaries.cpp
        )
    endif()
endforeach()
