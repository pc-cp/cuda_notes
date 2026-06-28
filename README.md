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


