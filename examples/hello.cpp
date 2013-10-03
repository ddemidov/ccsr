#include <iostream>
#include <ccsr.hpp>

int main(int argc, char *argv[]) {
    const int n = argc < 2 ? 4 : std::stoi(argv[1]);

    ccsr::matrix<double, int, int> A(n * n, n * n);

    std::vector<int>    row;
    std::vector<int>    col;
    std::vector<double> val;

    for(int i = 0, idx = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j, ++idx) {
            row.resize(0);
            col.resize(0);
            val.resize(0);

            row.push_back(0);

            if (i == 0 || j == 0 || i + 1 == n || j + 1 == n) {
                col.push_back(idx);
                val.push_back(1);
            } else {
                col.push_back(idx - n);
                col.push_back(idx - 1);
                col.push_back(idx + 0);
                col.push_back(idx + 1);
                col.push_back(idx + n);

                val.push_back(-1);
                val.push_back(-1);
                val.push_back( 4);
                val.push_back(-1);
                val.push_back(-1);
            }

            row.push_back(col.size());
            A.insert(idx, idx + 1, row.data(), col.data(), val.data());
        }
    }

    A.finish();

    std::cout
        << "Unique rows: " << A.unique_rows() << std::endl
        << "Nonzeros:    " << A.non_zeros()   << std::endl
        << "Compression: " << A.compression() << std::endl
        << std::endl;

    if (n < 20) {
        for(size_t row = 0; row < n * n; ++row) {
            for(auto i = A.begin(row); i != A.end(row); ++i)
                std::cout
                    << boost::get<1>(*i) << "("
                    << boost::get<0>(*i) << ") ";
            std::cout << std::endl;
        }
    }

    auto B = transp(A);

    std::cout
        << "Unique rows: " << B.unique_rows() << std::endl
        << "Nonzeros:    " << B.non_zeros()   << std::endl
        << "Compression: " << B.compression() << std::endl
        << std::endl;

    if (n < 20) {
        for(size_t row = 0; row < n * n; ++row) {
            for(auto i = B.begin(row); i != B.end(row); ++i)
                std::cout
                    << boost::get<1>(*i) << "("
                    << boost::get<0>(*i) << ") ";
            std::cout << std::endl;
        }
    }

}
