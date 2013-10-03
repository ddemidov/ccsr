### Lossy compression for CSR sparse matrix format

Each row is stored as a pointer to a table of unique matrix rows, where
uniqueness is determined approximately.  This may result in significant storage
space savings in case matrix has regular structure (e.g. it represents a
Poisson problem in a domain with piecewise-constant or slowly changing
properties).  The precision loss is possible, but may not be important (e.g.
coefficients come from an experiment with incorporated observation error).
