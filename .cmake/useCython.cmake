# Define a function to create Cython modules.
#
# For more information on the Cython project, see http://cython.org/.
# "Cython is a language that makes writing C extensions for the Python language
# as easy as Python itself."
#
# This file defines a CMake function to build a Cython Python module.
# To use it, first include this file.
#
#   include( useCython )
#
# Then call cython_add_module to create a module.
#
#   cython_add_module( <module_name> <src1> <src2> ... <srcN> )
#
# Where <module_name> is the name of the resulting Python module and
# <src1> <src2> ... are source files to be compiled into the module, e.g. *.pyx,
# *.py, *.cxx, etc.  A CMake target is created with name <module_name>.  This can
# be used for target_link_libraries(), etc.
#
# The sample paths set with the CMake include_directories() command will be used
# for include directories to search for *.pxd when running the Cython complire.
#
# Cache variables that effect the behavior include:
#
#  CYTHON_ANNOTATE
#  CYTHON_NO_DOCSTRINGS
#  CYTHON_FLAGS
#
# See also FindCython.cmake

#=============================================================================
# Copyright 2011 Kitware, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

# Configuration options.
set( CYTHON_ANNOTATE OFF
  CACHE BOOL "Create an annotated .html file when compiling *.pyx." )
set( CYTHON_NO_DOCSTRINGS OFF
  CACHE BOOL "Strip docstrings from the compiled module." )
set( CYTHON_FLAGS "" CACHE STRING
  "Extra flags to the cython compiler." )
mark_as_advanced( CYTHON_ANNOTATE CYTHON_NO_DOCSTRINGS CYTHON_FLAGS )

# Create a *.cxx file from a *.pyx file.
# Input the generated file basename.  The generate file will put into the variable
# placed in the "generated_file" argument. Finally all the *.py and *.pyx files.
function( compile_pyx _name generated_file )

  set( pyx_locations "" )

  foreach( pyx_file ${ARGN} )
    # Get the include directories.
    get_source_file_property( pyx_location ${pyx_file} LOCATION )
    get_filename_component( pyx_path ${pyx_location} PATH )
    get_directory_property(cmake_include_directories DIRECTORY ${pyx_path} INCLUDE_DIRECTORIES)
    list( APPEND pyx_locations "${pyx_location}" )
  endforeach() # pyx_file

  # Set additional flags.
  set(cython_args "")
  if( CYTHON_ANNOTATE )
    list(APPEND cython_args "--annotate" )
  endif()

  if( CYTHON_NO_DOCSTRINGS )
    list(APPEND cython_args "--no-docstrings")
  endif()

  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug" OR
     "${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    set(APPEND cython_args "--gdb")
  endif()

  list(APPEND cython_args "-${Python_VERSION_MAJOR}")

  # Include directory arguments.
  list(REMOVE_DUPLICATES cmake_include_directories)
  foreach(_include_dir ${cmake_include_directories})
    list(APPEND cython_args "-I${_include_dir}")
  endforeach()

  # Determining generated file name.
  set(_generated_file ${CMAKE_CURRENT_BINARY_DIR}/${_name}.cxx)
  set_source_files_properties( ${_generated_file} PROPERTIES GENERATED TRUE )
  set( ${generated_file} ${_generated_file} PARENT_SCOPE )

  # Add the command to run the compiler.
  add_custom_command( OUTPUT ${_generated_file}
    COMMAND ${CYTHON_EXECUTABLE}
    ARGS --cplus ${cython_args} ${CYTHON_FLAGS}
    --output-file  ${_generated_file} ${pyx_locations}
    DEPENDS ${pyx_locations}
    IMPLICIT_DEPENDS CXX
    COMMENT "Compiling Cython CXX source for ${_name}..."
    )
endfunction()

# cython_add_module( <name> src1 src2 ... srcN )
# Build the Cython Python module.
function( cython_add_module _name )
  set( pyx_module_sources "" )
  set( other_module_sources "" )
  foreach( _file ${ARGN} )
    if( ${_file} MATCHES ".*\\.py[x]?$" )
      list( APPEND pyx_module_sources ${_file} )
    else()
      list( APPEND other_module_sources ${_file} )
    endif()
  endforeach()
  set( CYTHON_FLAGS ${CYTHON_FLAGS} -X embedsignature=True)
  compile_pyx( ${_name} generated_file ${pyx_module_sources} )
  Python_add_library( ${_name} MODULE ${generated_file} ${other_module_sources} )
  if( APPLE )
    set_target_properties( ${_name} PROPERTIES LINK_FLAGS "-undefined dynamic_lookup" )
  endif()
  # ignore overflow warnings caused by Python's implicit conversions
  set_property( SOURCE ${generated_file}
                PROPERTY COMPILE_OPTIONS -Wno-overflow APPEND )
  # ignore Numpy deprecated API warning
  # ignore warnings for using the #warning extension directive
  target_compile_options( ${_name} PRIVATE -Wno-cpp -Wno-pedantic)
endfunction()
