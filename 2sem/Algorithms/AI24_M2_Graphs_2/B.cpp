#include <iostream>
#include <vector>

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);
    uint32_t n;
    std::cin >> n;

    std::vector<std::vector<int64_t>> v(n, std::vector<int64_t>(n, 0));

    for (uint32_t i = 0;i < n;++i)
        for (uint32_t j = 0;j < n;++j)
            std::cin >> v[i][j]; 

    for (uint32_t k = 0;k < n;++k)
        for (uint32_t i = 0;i < n;++i)
            for (uint32_t j = 0;j < n;++j)
                v[i][j] = std::min(v[i][j], v[i][k] + v[k][j]);

    for (uint32_t i = 0;i < n;++i) {
        for (uint32_t j = 0;j < n;++j)
            std::cout << v[i][j] << " "; 
        std::cout << std::endl;
    }
}