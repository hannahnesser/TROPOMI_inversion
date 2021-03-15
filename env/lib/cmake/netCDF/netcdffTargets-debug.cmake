#----------------------------------------------------------------
# Generated CMake target import file for configuration "DEBUG".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "netCDF::netcdff" for configuration "DEBUG"
set_property(TARGET netCDF::netcdff APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(netCDF::netcdff PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_DEBUG "netCDF::netcdf"
  IMPORTED_LOCATION_DEBUG "/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/env/lib/libnetcdff.7.0.0.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libnetcdff.7.dylib"
  )

list(APPEND _IMPORT_CHECK_TARGETS netCDF::netcdff )
list(APPEND _IMPORT_CHECK_FILES_FOR_netCDF::netcdff "/Users/hannahnesser/Documents/Harvard/Research/TROPOMI_Inversion/env/lib/libnetcdff.7.0.0.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
