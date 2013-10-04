### Lossy compression for CSR sparse matrix format

Each row is stored as a pointer to a table of unique matrix rows, where
uniqueness is determined approximately.  This may result in significant storage
space savings in case matrix has regular structure (e.g. it represents a
Poisson problem in a domain with piecewise-constant or slowly changing
properties).  The precision loss is possible, but may not be important (e.g.
coefficients come from an experiment with incorporated observation error).

The `perf_benchmark` example compares performance of matrix operations with
compressed and uncompressed CSR formats. The matrix tested is discretization of
2D Poisson problem with constant coefficients. Here is profile output for
6144x6144 matrix:

    [Profile:          31.247 sec.] (100.00%)
    [  CCSR:           17.745 sec.] ( 56.79%)
    [    assemble:      2.808 sec.] (  8.99%)
    [    multiply:      9.343 sec.] ( 29.90%)
    [    transpose:     5.576 sec.] ( 17.84%)
    [  CSR:            13.503 sec.] ( 43.21%)
    [   self:           1.468 sec.] (  4.70%)
    [    assemble:      1.208 sec.] (  3.87%)
    [    multiply:      8.440 sec.] ( 27.01%)
    [    transpose:     2.386 sec.] (  7.64%)

As you can see, operations with compressed matrices are no more than twice
slower than their uncompressed counterparts. At the same time, uncompressed
test required 12.5 GB of RAM, while compressed matrices were able to fit into
0.7 GB.
