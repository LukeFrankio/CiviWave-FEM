# DetectCompilerABI.cmake
# Computes a simple ABI fingerprint from compiler ID, version, and critical flags.
# Intended to be used to key dependency caches and trigger rebuilds when ABI changes.

function(cw_compute_abi_fingerprint OUT_VAR)
    set(_id   "${CMAKE_CXX_COMPILER_ID}")
    set(_ver  "${CMAKE_CXX_COMPILER_VERSION}")
    set(_std  "${CMAKE_CXX_STANDARD}")
    set(_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE}}")
    string(SHA1 _hash "${_id}|${_ver}|c++${_std}|${_flags}")
    set(${OUT_VAR} "${_hash}" PARENT_SCOPE)
endfunction()

# Emit the fingerprint (optional)
cw_compute_abi_fingerprint(_cw_abi)
message(STATUS "Compiler ABI fingerprint: ${_cw_abi}")
