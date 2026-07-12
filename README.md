# PMPP

## chapter 3 - multidimensional grids and data

viewpoints from l get:

1. multi grid served by multi data, like matrix multiplication. and threads per block have limit, like 1024, but blocks per grid no number limit.
	"page 49: the total size of a block in current CUDA systems is limited to 1024 threads. These threads can be distributed across the three dimensions in any way as long as
	the total number of threads does not exceed 1024" <-- blockDim.x * blockDim.y * blockDim.z <= 1024

	
2. programmers have responsibility to use flat index to access multi dimensions data
	"page 53:  the ANSI C standard on the basis of which CUDA C was developed requires the number of columns in array to be known at compile tiem for array to be accessed as a 2D array. Unfortunately, this information is not known at compile time for dynamically allocated arrays. In fact, part of the reason why one uses dynamically allocated arrays is to allow the size and dimensions of these arrays to vary according to the data size at runtime. Thus the information on the numberof columns in a dynamically allocated 2D array is not known at compile time by design. As a result, programmers need to explicitly linearize, or "flatten," a dynamically allocated 2D array into an equivalent 1D array in the current CUDA C.


## chapter 4 - compute architecture and scheduling

viewpoints from l get:

1. architecture of GPU: compute architecture and memory architecture. the first have logic: grid, block, warp and thread, and hardware: GPU, SM, processing block and core. memory architecture belong chapter 5, Memory Architecture and Data Locality and chapter 6, Performance Considerations.

2. block scheduling have transparent scalability, so same program can run different hardware that processing different resource. at the same time, a SM can accept multiple blocks, and of course, different compute capability have maxBlocksPerSM limits should care.

3. barrier sync on block level and warp level should know we shouldn't put \__syncthreads in warp divergence. and know warp is a unit of thread scheduling. one processing block in SM can manage many warp, one more thing is occupancy: threads run in SM / max threads that SM accept. high occupant means more possibility to latency tolerance: one warp wait long latency operation, so other warp been scheduled by processing block.

4. SM have limit resource, so maxThreadsPerBlock,  maxThreadsPerSM, maxblocksPerSM, registersPerSM, and Shread Memory per SM/Block, we should be care when we run execute configure parameter.


## chapter 5 - Memory architecture and data locality
viewpoints from l get:

1. this chapter origin from compute intensity -> roofline model -> how improve compute intensity for the same problem? because compute intensity = FLOPs / total bytes moved across the memory system ( both reads and writes ). can we can reuse data on on-chip memory to reduce Denominator. so Tiling technique that use SMEM to reduce access off-chip: GMEM.

2. but SMEM is source in SM, so SMEM possible influence occupancy: if one block use more SMEM, blocks reside on SM will decrease.

3. Tiling technique separate one operation to multiple phases, each phase, threads in block cooperate to load data that threads in block use. then this phase, all threads in block just use data that loaded (**Locality**). then wait threads in block read finish, move next phase to load GMEM until finish their job. so each thread load times decrease. but one operation been separate multiple phases.

-----------

some viewpoints to supplement:

1. memory type: load/store registers produce less instructions, but for Cache or memory: SMEM, GMEM, need load/store operation. Since the processer can fetch and execute only a limited number of instructions per clock cycle.

2. In modern computers the energy that is consumed for accessing a value from the register file is at least an order of magnitude lower than for accessing a value from the global memory.

3. use register as soon as possible? exceeds the limit of the source lead occupancy decrease.

4. Automatic array variables are not stored in registers. <-- some exceptions: The compiler may decide to store an automatic array into registers if all accesses are done with constant index values.

## chapter 6 - Performance considerations
viewpoints from l get:

1. this is the last chapter of introduction, next chapter will use some patterns to practice these mindsets.

2. optimization is not guess work, it based on your hardware and analyze program.

3. until, we know some optimizes, from thread to block, and from on-chip memory to off-chip memory: 
	- SIMD execute and avoid divergence
	- occupancy to hide latency
	- tiling technique to reuse data and improve compute intensity
	- GMEM access coalescing to reduce access transaction, (corner turning to achieve memory coalescing).
	- thread coarsening to reduce price of fake parallel.
	- *privatization to less contention and serialzation of atomic updates.

--------- 
some viewpoints to supplement:
1. DRAM bursting is a form of parallel organization: Multiple locations are accessed in the DRAM core array in parallel. However, bursting alone is not sufficient to realize the level of DRAM access bandwidth required by modern processes. it employ two more forms of parallel organization: banks and channels.
 
2. In general, if the ratio of the cell array access latency and data transfer time is R, we need to have at least R + 1 banks if we hope to fully utilize the data transfer bandwidth of the channel bus. In general, the number of banks connected to each channel bus needs to be larger than R for two reasons. 
	- having more banks reduces the probability of multiple simultaneous accesses targeting the same bank: bank conflict.
	- the size of each cell array is set to achieve reasonable latency and manufacturability. This limits the number of cells that each bank can provide. One may need many banks just to be able to support the memory size that is required.

3. The disadvantage of parallel work at the finest granularity comes when there is a "price" to be paid for parallel that work. such as redundant loading of data by different thread blocks, redundant work, sync overhead, and others. When the threads are executed in parallel by the HW, this price is often worth paying. If the HW ends up serializing the work as a result of insufficient resources, then this price has been paid unnecessary: assigning each thread multiple units of work, thread coarsening.

| Optimization | Benefit to compute cores | Benefit to memory | Strategies |
|---|---|---|---|
| Maximizing occupancy | More work to hide pipeline latency | More parallel memory accesses to hide DRAM latency | Tuning usage of SM resources such as threads per block, SMEM per block, and so on |
| Enabling coalesced GMEM access | Fewer pipeline stalls waiting for GMEM access | Less GMEM traffic and better utilization of bursts/cache lines | Transfer between GMEM and SMEM in a coalesced manner and perform uncoalesced accesses in SMEM (e.g., corner turning); rearranging the mapping of threads to data; rearranging the layout of the data |
| Minimizing control divergence | High SIMD efficiency | - | Rearranging the mapping of threads to data; rearranging the layout of the data |
| Tiling of reused data | Fewer pipeline stalls waiting for GMEM | Less GMEM traffic | Placing data that is reused within a block in SMEM or registers so that it is transferred between GMEM and the SM only once |
| Privatization (covered later) | Fewer pipeline stalls waiting for atomic updates | Less contention and serialization of atomic updates | Applying partial updates to a private copy of the data and then updating the universal copy when done |
| Thread coarsening | Less redundant work, divergence, or synchronization | Less redundant GMEM traffic | Assigning multiple units of parallelism to each thread to reduce the price of parallelism when it is incurred unnecessarily |