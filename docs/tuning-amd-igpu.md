# AMD iGPU Tuning Notes

This guide centralizes heuristics for squeezing performance out of the reference
AMD Radeon™ integrated GPU.

## Subgroup Configuration

- Force `requiredSubgroupSize = 64` for every compute pipeline that uses
  subgroup operations.
- Target workgroup sizes of 128 or 256 threads to keep occupancy high without
  starving registers.

## Memory Strategy

- Favor struct-of-arrays layouts and keep hot vectors in FP32 to minimize
  bandwidth pressure.
- Use descriptor buffer regions to shard allocations that exceed the 2 GB device
  limit; avoid rebinding descriptor sets every frame.
- Prefer recomputing small derived quantities (like the B-matrix) when the cost
  is lower than fetching from memory.

## Mixed Precision

- Keep reduction kernels in FP64 to preserve solver robustness while allowing
  FP32 for the bulk of vector math.
- Monitor iteration counts; a sudden increase usually signals precision issues
  or a preconditioner regression.

## Profiling Workflow

1. Use Vulkan timestamps to bracket every compute pass and record the data in
   the per-frame telemetry log.
2. Capture GPU traces with RGP when iteration counts spike or frame times swing.
3. Track compiler flags and shader revisions in [`docs/decisions.yaml`](decisions.yaml)
   whenever tuning changes behaviour across driver updates.
