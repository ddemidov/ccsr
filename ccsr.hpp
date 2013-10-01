#ifndef CCSR_HPP
#define CCSR_HPP

/**
 * \file   ccsr.hpp
 * \author Denis Demidov <ddemidov@ksu.ru>
 * \brief  Compressed CSR format
 *
 * Lossy compression for CSR format. Stores unique matrix rows, where
 * uniqueness is determined approximately. If the precision loss is not
 * important (e.g. coefficients come from an experiment with incorporated
 * observation error), it may result in significant storage savings for large
 * matrices.
 */


#include <vector>
#include <deque>
#include <type_traits>
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

template <typename T>
struct shift {
    T s;
    shift(T s) : s(s) {}

    T operator()(T v) const {
        return v + s;
    }
};


template <
    typename val_t = double,
    typename row_t = size_t,
    typename col_t = ptrdiff_t,
    typename idx_t = row_t
    >
class matrix;

template <
    typename val_t = double,
    typename row_t = size_t,
    typename col_t = ptrdiff_t,
    typename idx_t = row_t
    >
class matrix_builder {
    static_assert(std::is_signed<col_t>::value, "Column type should be signed!");

    private:
        std::deque< idx_t >                    idx;
        std::deque< row_t >                    row;
        std::deque< std::tuple<col_t, val_t> > val;

        struct row_hasher {
            template <class Range>
            size_t operator()(const Range &r) const {
                using std::get;
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
                using std::get;
                using boost::get;

                auto i1 = get<0>(r1);
                auto e1 = get<1>(r1);
                auto i2 = get<0>(r2);
                auto e2 = get<1>(r2);

                if (e1 - i1 != e2 - i2) return false;

                static const val_t eps = 1e-4;

                for(; i1 != e1; ++i1, ++i2) {
                    if (get<0>(*i1) != get<0>(*i2))
                        return false;

                    if (fabs(get<1>(*i1) - get<1>(*i2)) > eps)
                        return false;
                }

                return true;
            }
        } hasher;

        typedef typename std::deque<
                    std::tuple< col_t, val_t >
                >::const_iterator row_iterator;

        typedef std::tuple< row_iterator, row_iterator, size_t > row_range;

        boost::unordered_set<row_range, row_hasher, row_hasher> index;

    public:
        matrix_builder() : index(1979, hasher, hasher) {
            row.push_back(0);
        }

        void insert(size_t row_begin, size_t row_end,
                const row_t *r, const col_t *c, const val_t *v)
        {
            for(col_t i = row_begin, j = 0; i < row_end; ++i, ++j) {
                auto range = boost::make_tuple(
                        boost::make_zip_iterator(
                            boost::make_tuple(
                                boost::make_transform_iterator(
                                    c + r[j], shift<col_t>(-i)
                                    ),
                                v + r[j])
                            ),
                        boost::make_zip_iterator(
                            boost::make_tuple(
                                boost::make_transform_iterator(
                                    c + r[j + 1], shift<col_t>(-i)
                                    ),
                                v + r[j + 1])
                            )
                        );

                auto pos = index.find(range, hasher, hasher);

                if (pos == index.end()) {
                    idx.push_back(index.size());

                    size_t start = val.size();

                    for(size_t k = r[j]; k < r[j+1]; ++k)
                        val.push_back( std::make_pair(c[k] - i, v[k]) );

                    index.insert( std::make_tuple(val.begin() + start, val.end(), index.size()) );

                    row.push_back(val.size());
                } else {
                    idx.push_back(std::get<2>(*pos));
                }
            }
        }

        size_t unique_rows() const {
            return index.size();
        }

        friend class matrix<val_t, row_t, col_t, idx_t>;
};


template <typename val_t, typename row_t, typename col_t, typename idx_t>
class matrix {
    static_assert(std::is_signed<col_t>::value, "Column type should be signed!");

    private:
        size_t rows, cols;

        std::vector< idx_t > idx;
        std::vector< row_t > row;
        std::vector< col_t > col;
        std::vector< val_t > val;

    public:
        typedef boost::transform_iterator<
                    shift<col_t>,
                    typename std::vector<col_t>::const_iterator
                > column_iterator;

        typedef boost::tuple<
                    column_iterator, typename std::vector<val_t>::const_iterator
                > const_iterator_tuple;

        typedef boost::zip_iterator<const_iterator_tuple>
                const_row_iterator;

        /// Constructor.
        matrix(size_t rows, size_t cols,
                const matrix_builder<val_t, row_t, col_t, idx_t> &builder
              )
            : rows(rows), cols(cols),
              idx(builder.idx.begin(), builder.idx.end()),
              row(builder.row.begin(), builder.row.end())
        {
            col.reserve(builder.val.size());
            val.reserve(builder.val.size());

            for(auto v = builder.val.begin(); v != builder.val.end(); ++v) {
                col.push_back(std::get<0>(*v));
                val.push_back(std::get<1>(*v));
            }
        }

        /// Begin boost::zip_iterator to columns/values for a given row.
        const_row_iterator begin(size_t i) const {
            assert(i < rows);

            return const_row_iterator(
                    const_iterator_tuple(
                        column_iterator(col.begin() + row[idx[i]], shift<col_t>(i)),
                        val.begin() + row[idx[i]]
                        )
                    );
        }

        /// End boost::zip_iterator to columns/values for a given row.
        const_row_iterator end(size_t i) const {
            assert(i < rows);

            return const_row_iterator(
                    const_iterator_tuple(
                        column_iterator(col.begin() + row[idx[i] + 1], shift<col_t>(i)),
                        val.begin() + row[idx[i] + 1]
                        )
                    );
        }
};

} // namespace ccsr

#endif
