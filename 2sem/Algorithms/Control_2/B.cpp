#include <iostream>
#include <vector>
#include <limits>

#define MINUS_INF std::numeric_limits<int64_t>::min()

int64_t get(int64_t t) {
    return t == MINUS_INF ? MINUS_INF : t;
}

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);
    int64_t n;
    n = 8;
    std::vector<std::vector<uint32_t>> v(n, std::vector<uint32_t>(n, 0));

    for (int64_t i = 0; i < n;++i) {
        for (int64_t j = 0;j < n;++j) {
            std::cin >> v[i][j];
        }
    }

    for (int64_t i = n - 1;i >= 0;--i) {
        for (int64_t j = 0;j < n;++j) {
            if (i == n - 1 && j == 0) {
                continue;
            }
            else if (i == n - 1 && j > 0) {
                v[i][j] += v[i][j - 1];
            }
            else if (i < n - 1 && j == 0) {
                v[i][j] += v[i + 1][j];
            }
            else {
                v[i][j] += std::min(
                    v[i + 1][j],
                    std::min(
                        v[i][j - 1],
                        v[i + 1][j - 1]
                    )
                );
            }
        }
    }
    std::cout << v[0][n - 1] << std::endl;
}