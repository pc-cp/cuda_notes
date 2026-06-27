# PMPP

## chapter 3 - multidimensional grids and data

viewpoints from l get:

1. multi grid served by multi data, like matrix multiplication. and threads per block have limit, like 1024, but blocks per grid no number limit.
	"page 49: the total size of a block in current CUDA systems is limited to 1024 threads. These threads can be distributed across the three dimensions in any way as long as
	the total number of threads does not exceed 1024" <-- blockDim.x * blockDim.y * blockDim.z <= 1024

	
2. programmers have responsibility to use flat index to access multi dimensions data
	"page 53:  the ANSI C standard on the basis of which CUDA C was developed requires the number of columns in array to be known at compile tiem for array to be accessed as a 2D array. Unfortunately, this information is not known at compile time for dynamically allocated arrays. In fact, part of the reason why one uses dynamically allocated arrays is to allow the size and dimensions of these arrays to vary according to the data size at runtime. Thus the information on the numberof columns in a dynamically allocated 2D array is not known at compile time by design. As a result, programmers need to explicitly linearize, or "flatten," a dynamically allocated 2D array into an equivalent 1D array in the current CUDA C.
