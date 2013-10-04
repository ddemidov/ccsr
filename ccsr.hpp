#ifndef CCSR_HPP
#define CCSR_HPP

/*
The MIT License

Copyright (c) 2013 Denis Demidov <ddemidov@ksu.ru>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/**
 * \file   ccsr.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Lossy compression for CSR sparse matrix format.
 *
 * Each row is stored as a pointer to a table of unique matrix rows, where
 * uniqueness is determined approximately.  This may result in significant
 * storage space savings in case matrix has regular structure (e.g. it
 * represents a Poisson problem in a domain with piecewise-constant or slowly
 * changing properties).  The precision loss is possible, but may not be
 * important (e.g.  coefficients come from an experiment with incorporated
 * observation error).
 */


#include <vector>
#include <deque>
#include <type_traits>
#include <memory>
#include <functional>
#include <numeric>
#include <cassert>

#include <boost/unordered_set.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>

namespace ccsr {

namespace detail {

template <typename T>
struct hash_impl {
    static inline size_t get(T v) {
        return std::hash<T>()(v);
    }
};

template <>
struct hash_impl<double> {
    static inline size_t get(double v) {
        static const uint64_t mask = 0xffffffffff000000;
                                    // eeefffffffffffff;
        // Zero-out least significant bits and hash as uint64_t:
        return std::hash<uint64_t>()(*reinterpret_cast<const uint64_t*>(&v) & mask);
    }
};

template <>
struct hash_impl<float> {
    static inline size_t get(float v) {
        static const uint32_t mask = 0xfffffc00;
                                    // eeffffff;
        // Zero-out least significant bits and hash as uint64_t:
        return std::hash<uint32_t>()(*reinterpret_cast<const uint32_t*>(&v) & mask);
    }
};

}

template <class T>
inline size_t hash(const T &v) {
    return detail::hash_impl<T>::get(v);
}

/// Lossy compression for CSR sparse matrix format.
/**
 * Each row is stored as a pointer to a table of unique matrix rows, where
 * uniqueness is determined approximately.  This may result in significant
 * storage space savings in case matrix has regular structure (e.g. it
 * represents a Poisson problem in a domain with piecewise-constant or slowly
 * changing properties).  The precision loss is possible, but may not be
 * important (e.g.  coefficients come from an experiment with incorporated
 * observation error).
*/
template <
    typename val_t = double,
    typename row_t = size_t,
    typename col_t = ptrdiff_t,
    typename idx_t = row_t
    >
class matrix {
    static_assert(std::is_signed<col_t>::value, "Column type should be signed!");

    private:
        size_t nrows, ncols, nnz;
        val_t  eps;

        std::vector< idx_t > idx;
        std::vector< row_t > row;
        std::vector< col_t > col;
        std::vector< val_t > val;

        // Returns operand incremented by a given value.
        struct shift {
            typedef col_t result_type;

            col_t s;

            shift(col_t s) : s(s) {}

            col_t operator()(col_t v) const {
                return v + s;
            }
        };

        // Extracts and stores unique rows.
        struct builder_t {
            std::deque< idx_t > idx;
            std::deque< row_t > row;
            std::deque< col_t > col;
            std::deque< val_t > val;

            // Hashes and compares matrix rows.
            struct row_hasher {
                val_t eps;

                row_hasher(val_t eps) : eps(eps) {}

                template <class Range>
                size_t operator()(const Range &r) const {
                    using boost::get;

                    auto begin = get<0>(r);
                    auto end   = get<1>(r);

                    size_t h = hash(end - begin);

                    for(auto i = begin; i != end; ++i)
                        h ^= hash(get<0>(*i)) ^ hash(get<1>(*i));

                    return h;
                }

                template <class Range1, class Range2>
                bool operator()(const Range1 &r1, const Range2 &r2) const {
                    using boost::get;

                    auto i1 = get<0>(r1);
                    auto e1 = get<1>(r1);
                    auto i2 = get<0>(r2);
                    auto e2 = get<1>(r2);

                    if (e1 - i1 != e2 - i2) return false;

                    for(; i1 != e1; ++i1, ++i2) {
                        if (get<0>(*i1) != get<0>(*i2))
                            return false;

                        if (fabs(get<1>(*i1) - get<1>(*i2)) > eps)
                            return false;
                    }

                    return true;
                }
            } hasher;

            typedef boost::zip_iterator<
                        boost::tuple<
                            typename std::deque< col_t >::const_iterator,
                            typename std::deque< val_t >::const_iterator
                        >
                    > row_iterator;

            typedef boost::tuple< row_iterator, row_iterator, size_t > row_range;

            boost::unordered_set<row_range, row_hasher, row_hasher> index;

            builder_t(size_t nrows, val_t eps)
                : idx(nrows, 0), hasher(eps), index(1979, hasher, hasher)
            {
                // Artificial empty row:
                row.push_back(0);
                row.push_back(0);

                index.insert(
                        boost::make_tuple(
                            boost::make_zip_iterator(
                                boost::make_tuple(col.begin(), val.begin())
                                ),
                            boost::make_zip_iterator(
                                boost::make_tuple(col.end(), val.end())
                                ),
                            index.size()
                            )
                        );
            }

            void insert(col_t row_begin, col_t row_end,
                    const row_t *r, const col_t *c, const val_t *v)
            {
                for(col_t i = row_begin, j = 0; i < row_end; ++i, ++j) {
                    shift s(-i);

                    auto range = boost::make_tuple(
                            boost::make_zip_iterator(
                                boost::make_tuple(
                                    boost::make_transform_iterator(c + r[j], s),
                                    v + r[j])
                                ),
                            boost::make_zip_iterator(
                                boost::make_tuple(
                                    boost::make_transform_iterator(c + r[j+1], s),
                                    v + r[j+1])
                                )
                            );

                    auto pos = index.find(range, hasher, hasher);

                    if (pos == index.end()) {
                        idx[i] = index.size();

                        size_t start = val.size();

                        for(size_t k = r[j]; k < r[j+1]; ++k) {
                            col.push_back( c[k] - i );
                            val.push_back( v[k] );
                        }

                        index.insert(
                                boost::make_tuple(
                                    boost::make_zip_iterator(
                                        boost::make_tuple(
                                            col.begin() + start,
                                            val.begin() + start)
                                        ),
                                    boost::make_zip_iterator(
                                        boost::make_tuple(
                                            col.end(),
                                            val.end()
                                            )
                                        ),
                                    index.size()
                                    )
                                );

                        row.push_back(val.size());
                    } else {
                        idx[i] = boost::get<2>(*pos);
                    }
                }
            }
        };

        std::unique_ptr<builder_t> builder;

    public:
        typedef boost::zip_iterator<
                    boost::tuple<
                        boost::transform_iterator<
                            shift, typename std::vector<col_t>::const_iterator
                        >,
                        typename std::vector<val_t>::const_iterator
                    >
                > const_row_iterator;

        /// Constructor.
        matrix(size_t nrows, size_t ncols, val_t eps = 1e-5)
            : nrows(nrows), ncols(ncols), nnz(0), eps(eps),
              builder(new builder_t(nrows, eps))
        {
        }

        /// Store matrix slice.
        /**
         * May accept whole matrix, or just a slice of matrix rows.
         */
        void insert(col_t row_begin, col_t row_end,
                const row_t *r, const col_t *c, const val_t *v)
        {
            assert(builder);

            builder->insert(row_begin, row_end, r, c, v);
        }

        /// All rows have been processed; finalize the construction phase.
        void finish() {
            assert(builder);

            idx.assign(builder->idx.begin(), builder->idx.end());
            row.assign(builder->row.begin(), builder->row.end());
            col.assign(builder->col.begin(), builder->col.end());
            val.assign(builder->val.begin(), builder->val.end());

            builder.reset();

            nnz = std::accumulate(idx.begin(), idx.end(), static_cast<size_t>(0),
                    [&](size_t tot, size_t p) {
                        return tot + row[p + 1] - row[p];
                    });
        }

        /// Number of unique rows in the matrix.
        size_t unique_rows() const {
            return row.size() - 1;
        }

        /// Returns boost::zip_iterator to start of columns/values range for a given row.
        const_row_iterator begin(size_t i) const {
            assert(!builder && i < nrows);

            return boost::make_zip_iterator(
                    boost::make_tuple(
                        boost::make_transform_iterator(
                            col.begin() + row[idx[i]],
                            shift(i)
                            ),
                        val.begin() + row[idx[i]]
                        )
                    );
        }

        /// Returns boost::zip_iterator to end of columns/values range for a given row.
        const_row_iterator end(size_t i) const {
            assert(!builder && i < nrows);

            return boost::make_zip_iterator(
                    boost::make_tuple(
                        boost::make_transform_iterator(
                            col.begin() + row[idx[i] + 1],
                            shift(i)
                            ),
                        val.begin() + row[idx[i] + 1]
                        )
                    );
        }

        /// Number of rows.
        size_t rows() const {
            return nrows;
        }

        /// Number of cols.
        size_t cols() const {
            return ncols;
        }

        /// Number of nonzeros in the matrix.
        size_t non_zeros() const {
            assert(!builder);

            return nnz;
        }

        /// Compression ratio.
        double compression() const {
            assert(!builder);
            return 1.0 *
                (
                 sizeof(idx[0]) * idx.size() +
                 sizeof(row[0]) * row.size() +
                 sizeof(col[0]) * col.size() +
                 sizeof(val[0]) * val.size()
                ) /
                (
                    sizeof(row[0]) * (nrows + 1) +
                    sizeof(col[0]) * nnz        +
                    sizeof(val[0]) * nnz
                );
        }


        /// Matrix transpose.
        friend matrix transp(const matrix &A) {
            const col_t chunk_size = 1024;

            col_t lw = 0, rw = 0;
            for(size_t i = 0, n = A.unique_rows(); i < n; ++i) {
                for(row_t j = A.row[i]; j < A.row[i + 1]; ++j) {
                    lw = std::max(lw, -A.col[j]);
                    rw = std::max(rw,  A.col[j]);
                }
            }

            matrix T(A.ncols, A.nrows, A.eps);

            std::vector<row_t> row;
            std::vector<col_t> col;
            std::vector<val_t> val;

            for(col_t chunk = 0; chunk < A.nrows; chunk += chunk_size) {
                col_t row_start = std::max(chunk - rw,              static_cast<col_t>(0));
                col_t row_end   = std::min(chunk + chunk_size + lw, static_cast<col_t>(A.nrows));
                col_t chunk_end = std::min(chunk + chunk_size,      static_cast<col_t>(A.nrows));

                row.clear();
                col.clear();
                val.clear();

                row.resize(chunk_size + 1, 0);

                for(col_t i = row_start; i < row_end; ++i)
                    for(auto j = A.begin(i), e = A.end(i); j != e; ++j) {
                        col_t c = boost::get<0>(*j);
                        if (c >= chunk && c < chunk_end)
                            ++( row[c - chunk + 1] );
                    }

                std::partial_sum(row.begin(), row.end(), row.begin());

                col.resize(row.back());
                val.resize(row.back());

                for(size_t i = row_start; i < row_end; ++i) {
                    for(auto j = A.begin(i), e = A.end(i); j != e; ++j) {
                        col_t c = boost::get<0>(*j);
                        val_t v = boost::get<1>(*j);

                        if (c >= chunk && c < chunk_end) {
                            row_t head = row[c - chunk]++;

                            col[head] = i;
                            val[head] = v;
                        }
                    }
                }

                std::rotate(row.begin(), row.end() - 1, row.end());
                row[0] = 0;

                T.insert(chunk, chunk_end, row.data(), col.data(), val.data());
            }

            T.finish();

            return T;
        }

        /// Matrix-matrix product.
        friend matrix prod(const matrix &A, const matrix &B) {
            matrix C(A.nrows, B.ncols, std::max(A.eps, B.eps));

            std::vector<col_t> marker(B.ncols, -1);

            row_t row[2] = {0, 0};
            std::vector<col_t> col;
            std::vector<val_t> val;

            for(size_t ia = 0; ia < A.nrows; ++ia) {
                col.clear();
                val.clear();

                for(auto ja = A.begin(ia), ea = A.end(ia); ja != ea; ++ja) {
                    col_t ca = boost::get<0>(*ja);
                    val_t va = boost::get<1>(*ja);

                    for(auto jb = B.begin(ca), eb = B.end(ca); jb != eb; ++jb) {
                        col_t cb = boost::get<0>(*jb);
                        val_t vb = boost::get<1>(*jb);

                        if (marker[cb] < 0) {
                            marker[cb] = col.size();
                            col.push_back(cb);
                            val.push_back(va * vb);
                        } else {
                            val[marker[cb]] += va * vb;
                        }
                    }
                }

                for(auto c = col.begin(); c != col.end(); ++c)
                    marker[*c] = -1;

                row[1] = col.size();

                C.insert(ia, ia + 1, row, col.data(), val.data());
            }

            C.finish();
            return C;
        }

};

} // namespace ccsr

#endif
