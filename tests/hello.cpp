#include <iostream>
#include <ccsr.hpp>

int main() {
    const int n = 4;

    ccsr::matrix_builder<double, int, int> B;

    for(int i = 0, idx = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j, ++idx) {
            std::vector<int>    row = {0};
            std::vector<int>    col;
            std::vector<double> val;

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
            B.insert(idx, idx + 1, row.data(), col.data(), val.data());
        }
    }

    std::cout << B.unique_rows() << std::endl;

    ccsr::matrix<double, int, int> A(n*n, n*n, B);

    for(size_t row = 0; row < n * n; ++row) {
        for(auto i = A.begin(row); i != A.end(row); ++i)
            std::cout
                << std::scientific << std::showpos
                << boost::get<1>(*i) << "("
                << boost::get<0>(*i) << ") ";
        std::cout << std::endl;
    }
}
