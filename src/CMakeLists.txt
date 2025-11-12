add_library(cwf_core
	config/config.cpp
	mesh/pack.cpp
	mesh/mesh.cpp
	mesh/preprocess.cpp
	gpu/sharding.cpp
	gpu/upload.cpp
	gpu/pcg.cpp
	physics/loads.cpp
	physics/newmark.cpp
	physics/solver.cpp
	gpu/vulkan_context.cpp
)

set(CWF_SHADER_SOURCES
	${CMAKE_SOURCE_DIR}/src/shaders/ke_apply_element.slang
	${CMAKE_SOURCE_DIR}/src/shaders/ke_gather_node.slang
	${CMAKE_SOURCE_DIR}/src/shaders/pcg_precondition.slang
	${CMAKE_SOURCE_DIR}/src/shaders/pcg_axpy.slang
	${CMAKE_SOURCE_DIR}/src/shaders/pcg_dot_partials.slang
	${CMAKE_SOURCE_DIR}/src/shaders/pcg_reduce_final.slang
	${CMAKE_SOURCE_DIR}/src/shaders/newmark_predictor.slang
	${CMAKE_SOURCE_DIR}/src/shaders/newmark_update.slang
	${CMAKE_SOURCE_DIR}/src/shaders/derive_fields.slang
)

cw_slang_compile(CWF_SHADER_OUTPUTS ${CWF_SHADER_SOURCES})
add_dependencies(cwf_core shaders)
set_source_files_properties(${CWF_SHADER_SOURCES} PROPERTIES HEADER_FILE_ONLY TRUE)

set(CWF_SHADER_HEADERS
	${CMAKE_SOURCE_DIR}/src/shaders/common.slang
	${CMAKE_SOURCE_DIR}/src/shaders/reductions.slang
	${CMAKE_SOURCE_DIR}/src/shaders/bc.slang
)

target_sources(cwf_core PRIVATE ${CWF_SHADER_SOURCES} ${CWF_SHADER_HEADERS})
set_source_files_properties(${CWF_SHADER_HEADERS} PROPERTIES HEADER_FILE_ONLY TRUE)

target_include_directories(cwf_core
	PUBLIC
		$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
		$<INSTALL_INTERFACE:include>
)

target_link_libraries(cwf_core
	PUBLIC
		yaml-cpp
		Vulkan::Vulkan
		VulkanMemoryAllocator
)

target_compile_features(cwf_core PUBLIC cxx_std_26)
