add_library(cwf_core
	config/config.cpp
	mesh/pack.cpp
	mesh/mesh.cpp
	mesh/preprocess.cpp
	gpu/sharding.cpp
	gpu/upload.cpp
	gpu/buffers.cpp
	gpu/device_buffers.cpp
	gpu/newmark_stepper.cpp
	gpu/pcg.cpp
	gpu/instrumentation.cpp
	physics/loads.cpp
	physics/newmark.cpp
	physics/solver.cpp
	post/derived_fields.cpp
	post/vtu_writer.cpp
	post/probe_logger.cpp
	post/output_manager.cpp
	gpu/vulkan_context.cpp
)

if(BUILD_UI)
    target_sources(cwf_core PRIVATE ui/viewer.cpp)
endif()

set(CWF_SHADER_SOURCES
	${CMAKE_SOURCE_DIR}/src/shaders/ke_apply_element.slang
	${CMAKE_SOURCE_DIR}/src/shaders/ke_gather_node.slang
	${CMAKE_SOURCE_DIR}/src/shaders/pcg_precondition.slang
	${CMAKE_SOURCE_DIR}/src/shaders/pcg_axpy.slang
	${CMAKE_SOURCE_DIR}/src/shaders/pcg_dot_partials.slang
	${CMAKE_SOURCE_DIR}/src/shaders/pcg_reduce_final.slang
	${CMAKE_SOURCE_DIR}/src/shaders/pcg_enforce_dirichlet.slang
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
		# std::print/std::stacktrace (C++23 experimental goodness) live in libstdc++exp on MinGW 🎯
		$<$<AND:$<CXX_COMPILER_ID:GNU>,$<BOOL:${WIN32}>>:stdc++exp>
)

# Tracy profiler integration (Phase 11)
if(ENABLE_TRACY AND TARGET tracy_client)
	target_link_libraries(cwf_core PUBLIC tracy_client)
	target_compile_definitions(cwf_core PUBLIC CWF_ENABLE_TRACY=1)
	message(STATUS "Tracy profiler integration enabled for cwf_core")
else()
	target_compile_definitions(cwf_core PUBLIC CWF_ENABLE_TRACY=0)
endif()

if(BUILD_UI)
	target_link_libraries(cwf_core PUBLIC imgui glfw)
	if(WIN32)
		target_link_libraries(cwf_core PUBLIC opengl32)
	endif()

	add_executable(cwf_viewer_demo ui/viewer_demo.cpp)
	target_link_libraries(cwf_viewer_demo PRIVATE cwf_core)
	target_compile_features(cwf_viewer_demo PRIVATE cxx_std_26)

	set(CWF_VIEWER_VERT ${CMAKE_SOURCE_DIR}/src/shaders/viewer_mesh_vert.slang)
	set(CWF_VIEWER_FRAG ${CMAKE_SOURCE_DIR}/src/shaders/viewer_mesh_frag.slang)
	cw_slang_compile(CWF_VIEWER_VERT_SPV PROFILE vs_6_5 TARGET viewer_shaders ${CWF_VIEWER_VERT})
	cw_slang_compile(CWF_VIEWER_FRAG_SPV PROFILE ps_6_5 TARGET viewer_shaders ${CWF_VIEWER_FRAG})
	set_source_files_properties(${CWF_VIEWER_VERT} ${CWF_VIEWER_FRAG} PROPERTIES HEADER_FILE_ONLY TRUE)
	target_sources(cwf_core PRIVATE ${CWF_VIEWER_VERT} ${CWF_VIEWER_FRAG})
	add_dependencies(cwf_core viewer_shaders)
		# Ensure shaders are copied next to the viewer binary so users running the exe directly
		# (from a different working directory) can find the shaders via our runtime discovery.
		add_custom_command(TARGET cwf_viewer_demo POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:cwf_viewer_demo>/shaders"
			COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_BINARY_DIR}/shaders" "$<TARGET_FILE_DIR:cwf_viewer_demo>/shaders"
			COMMENT "Copying compiled shaders next to cwf_viewer_demo"
		)
endif()

target_compile_definitions(cwf_core PUBLIC CWF_ENABLE_UI=$<BOOL:${BUILD_UI}>)
target_compile_definitions(cwf_core PUBLIC CWF_SHADER_BUILD_DIR="${CMAKE_BINARY_DIR}/shaders")

target_compile_features(cwf_core PUBLIC cxx_std_26)
