add_library(cwf_core
	config/config.cpp
	mesh/mesh.cpp
	mesh/preprocess.cpp
	physics/loads.cpp
	physics/newmark.cpp
	physics/solver.cpp
	gpu/vulkan_context.cpp
)

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
