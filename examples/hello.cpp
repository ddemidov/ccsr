#include <iostream>
#include <chrono>
#include <ccsr.hpp>
#include <amgcl/profiler.hpp>

int main(int argc, char *argv[]) {
    const int n = argc < 2 ? 4 : std::stoi(argv[1]);

    amgcl::profiler<std::chrono::high_resolution_clock> prof;

    prof.tic("assemble");
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
    prof.toc("assemble");

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

    prof.tic("transpose");
    auto B = transp(A);
    prof.toc("transpose");

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

    prof.tic("multiply");
    auto C = prod(A, A);
    prof.toc("multiply");

    std::cout
        << "Unique rows: " << C.unique_rows() << std::endl
        << "Nonzeros:    " << C.non_zeros()   << std::endl
        << "Compression: " << C.compression() << std::endl
        << std::endl;

    if (n < 20) {
        for(size_t row = 0; row < n * n; ++row) {
            for(auto i = C.begin(row); i != C.end(row); ++i)
                std::cout
                    << boost::get<1>(*i) << "("
                    << boost::get<0>(*i) << ") ";
            std::cout << std::endl;
        }
    }

    std::cout << prof << std::endl;
}
