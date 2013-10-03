### Sparse matrix format with lossy compression

Lossy compression for CSR sparse matrix format. Each row is a pointer to a
table of unique matrix rows, where uniqueness is determined approximately. If
the precision loss is not important (e.g. coefficients come from an experiment
with incorporated observation error), it may result in significant storage
savings for large matrices.
