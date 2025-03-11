#include <iostream>
#include <vector>

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);
    uint64_t n;
    std::cin >> n;

    std::vector<std::vector<uint64_t>> v(n, std::vector<uint64_t>(n, 0));

    for (uint64_t i = 0;i < n;++i)
        for (uint64_t j = 0;j < n;++j) {
            int64_t t;
            std::cin >> t;
            v[i][j] = (t == -1) ? (uint32_t)-1 : t; 
        }

    for(uint64_t k = 0;k < n;++k)
        for(uint64_t i = 0;i < n;++i)
            for(uint64_t j = 0;j < n;++j)
                v[i][j] = std::min(v[i][j], v[i][k] + v[k][j]);

    std::vector<uint64_t> e(n, 0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            e[i] = std::max(e[i], v[i][j]);
        }
    }

    uint64_t r = (uint64_t)-1, d = 0;
    for (int i = 0; i < n; i++) {
        r = std::min(r, e[i]);
        d = std::max(d, e[i]);
    }

    std::cout << d << "\n" << r << std::endl;
}