# SlangCompile.cmake
# Utilities to compile Slang shaders (.slang) to SPIR-V (.spv) using slangc.

# cw_find_slangc(<OUT_VAR>) — attempts to locate the slangc executable.
# Preference order:
#  1) SLANGC environment variable (full path)
#  2) system PATH (find_program)
#  3) if present, the Slang build tree from FetchContent (last-resort)
function(cw_find_slangc OUT_VAR)
    # 1) Check environment override
    if(DEFINED ENV{SLANGC})
        set(_cand "$ENV{SLANGC}")
        # Allow SLANGC to be either the full path to the executable or a directory
        if(IS_DIRECTORY "${_cand}")
            if(WIN32)
                set(_try "${_cand}/slangc.exe")
            else()
                set(_try "${_cand}/slangc")
            endif()
            if(EXISTS "${_try}")
                set(${OUT_VAR} "${_try}" PARENT_SCOPE)
                return()
            endif()
        else()
            # If SLANGC points at a file, accept it if it exists
            if(EXISTS "${_cand}")
                set(${OUT_VAR} "${_cand}" PARENT_SCOPE)
                return()
            endif()
        endif()
    endif()
    # 2) Try system PATH
    find_program(_slangc_exe NAMES slangc)
    if(_slangc_exe)
        set(${OUT_VAR} "${_slangc_exe}" PARENT_SCOPE)
        return()
    endif()

    # 3) Last-resort: check if Slang was fetched and provides a built slangc
    if(DEFINED slang_SOURCE_DIR)
        # Common build tree location after FetchContent_MakeAvailable(slang)
        # Note: exact path can vary with Slang version/generator. Try best-effort locations.
        set(_possible
            "${slang_BINARY_DIR}/bin/slangc"
            "${slang_BINARY_DIR}/bin/Release/slangc"
            "${slang_BINARY_DIR}/bin/Debug/slangc"
            "${slang_BINARY_DIR}/slangc"
        )
        if(WIN32)
            # also try Windows executable names
            list(APPEND _possible
                "${slang_BINARY_DIR}/bin/slangc.exe"
                "${slang_BINARY_DIR}/bin/Release/slangc.exe"
                "${slang_BINARY_DIR}/bin/Debug/slangc.exe"
                "${slang_BINARY_DIR}/slangc.exe"
            )
        endif()
        foreach(p IN LISTS _possible)
            if(EXISTS "${p}")
                set(${OUT_VAR} "${p}" PARENT_SCOPE)
                return()
            endif()
        endforeach()
    endif()

    # Not found
    set(${OUT_VAR} "" PARENT_SCOPE)
endfunction()

# cw_slang_compile(<OUT_LIST_VAR> [PROFILE <stage_profile>] [TARGET <cmake_target>] SRC1.slang SRC2.slang ...)
# Compiles each .slang source to a corresponding .spv under ${CMAKE_BINARY_DIR}/shaders
# Returns the list of generated .spv files in OUT_LIST_VAR and wires them to the chosen aggregation target
# (default target name: 'shaders').
function(cw_slang_compile OUT_LIST)
    if(ARGC LESS 2)
        set(${OUT_LIST} "" PARENT_SCOPE)
        return()
    endif()

    set(_profile "cs_6_5")
    set(_target_name "shaders")
    set(_arg_list ${ARGN})
    set(_sources)

    while(_arg_list)
        list(GET _arg_list 0 _token)
        list(REMOVE_AT _arg_list 0)
        if(_token STREQUAL "PROFILE")
            if(NOT _arg_list)
                message(FATAL_ERROR "cw_slang_compile PROFILE keyword missing value")
            endif()
            list(GET _arg_list 0 _profile)
            list(REMOVE_AT _arg_list 0)
        elseif(_token STREQUAL "TARGET")
            if(NOT _arg_list)
                message(FATAL_ERROR "cw_slang_compile TARGET keyword missing value")
            endif()
            list(GET _arg_list 0 _target_name)
            list(REMOVE_AT _arg_list 0)
        else()
            list(APPEND _sources "${_token}")
        endif()
    endwhile()

    if(NOT _sources)
        set(${OUT_LIST} "" PARENT_SCOPE)
        return()
    endif()

    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/shaders")

    cw_find_slangc(_slangc)
    if(_slangc)
        message(STATUS "Found slangc: ${_slangc}")
    else()
        message(WARNING "slangc not found at configure time; shader compilation rules will be deferred and stub outputs will be created.")
    endif()

    set(_outputs)
    set(_include_dirs)
    foreach(src IN LISTS _sources)
        get_filename_component(_dir "${src}" DIRECTORY)
        list(APPEND _include_dirs "${_dir}")
    endforeach()
    list(REMOVE_DUPLICATES _include_dirs)

    set(_include_args)
    foreach(_idir IN LISTS _include_dirs)
        list(APPEND _include_args -I "${_idir}")
    endforeach()

    if(_slangc)
        message(STATUS "Slang include paths: ${_include_dirs}")
    endif()

    foreach(src IN LISTS _sources)
        get_filename_component(_name "${src}" NAME_WE)
        set(_out "${CMAKE_BINARY_DIR}/shaders/${_name}.spv")
        list(APPEND _outputs "${_out}")
        if(_slangc)
            add_custom_command(
                OUTPUT "${_out}"
                COMMAND ${_slangc} -target spirv -profile ${_profile} ${_include_args} -o "${_out}" "${src}"
                DEPENDS "${src}"
                VERBATIM
                COMMENT "Compiling Slang -> SPIR-V: ${src}"
            )
        else()
            # Create a harmless stub file to satisfy build dependencies when slangc isn't available.
            add_custom_command(
                OUTPUT "${_out}"
                COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/shaders"
                COMMAND ${CMAKE_COMMAND} -E touch "${_out}"
                DEPENDS "${src}"
                COMMENT "slangc not found: creating stub shader ${_out}"
            )
        endif()
    endforeach()

    if(NOT TARGET ${_target_name})
        add_custom_target(${_target_name} ALL DEPENDS ${_outputs})
    else()
        set(_extra_target "${_target_name}_${OUT_LIST}_extra")
        add_custom_target(${_extra_target} DEPENDS ${_outputs})
        add_dependencies(${_target_name} ${_extra_target})
    endif()
    set(${OUT_LIST} "${_outputs}" PARENT_SCOPE)
endfunction()
