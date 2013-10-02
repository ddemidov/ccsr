#ifndef CCSR_HPP
#define CCSR_HPP

/**
 * \file   ccsr.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Compressed CSR format composition.
 *
 * Lossy compression for CSR sparse matrix format. Stores unique matrix rows,
 * where uniqueness is determined approximately. If the precision loss is not
 * important (e.g. coefficients come from an experiment with incorporated
 * observation error), it may result in significant storage savings for large
 * matrices.
 */


#include <vector>
#include <deque>
#include <type_traits>
#include <memory>
#include <functional>
#include <exception>
#include <cassert>

#include <boost/unordered_set.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>

namespace ccsr {

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

template <class T>
inline size_t hash(const T &v) {
    return hash_impl<T>::get(v);
}

template <
    typename val_t = double,
    typename row_t = size_t,
    typename col_t = ptrdiff_t,
    typename idx_t = row_t
    >
class matrix {
    static_assert(std::is_signed<col_t>::value, "Column type should be signed!");

    private:
        size_t rows, cols;

        std::vector< idx_t > idx;
        std::vector< row_t > row;
        std::vector< col_t > col;
        std::vector< val_t > val;

        struct shift {
            typedef col_t result_type;

            col_t s;

            shift(col_t s) : s(s) {}

            col_t operator()(col_t v) const {
                return v + s;
            }
        };

        struct builder_t {
            std::deque< idx_t > idx;
            std::deque< row_t > row;
            std::deque< col_t > col;
            std::deque< val_t > val;

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

            builder_t(val_t eps = 1e-5f)
                : hasher(eps), index(1979, hasher, hasher)
            {
                row.push_back(0);
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
                        idx.push_back(index.size());

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
                        idx.push_back(boost::get<2>(*pos));
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
        matrix(size_t rows, size_t cols, val_t eps = 1e-5)
            : rows(rows), cols(cols), builder(new builder_t(eps))
        {
        }

        void insert(col_t row_begin, col_t row_end,
                const row_t *r, const col_t *c, const val_t *v)
        {
            builder->insert(row_begin, row_end, r, c, v);
        }

        size_t finish() {
            if (builder->idx.size() != rows) throw std::logic_error(
                    "Matrix construction is incomplete "
                    "(some rows are left untouched)"
                    );

            idx.assign(builder->idx.begin(), builder->idx.end());
            row.assign(builder->row.begin(), builder->row.end());
            col.assign(builder->col.begin(), builder->col.end());
            val.assign(builder->val.begin(), builder->val.end());

            builder.reset();

            return row.size() - 1;
        }

        /// Begin boost::zip_iterator to columns/values for a given row.
        const_row_iterator begin(size_t i) const {
            assert(!builder && i < rows);

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

        /// End boost::zip_iterator to columns/values for a given row.
        const_row_iterator end(size_t i) const {
            assert(!builder && i < rows);

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
};

} // namespace ccsr

#endif
