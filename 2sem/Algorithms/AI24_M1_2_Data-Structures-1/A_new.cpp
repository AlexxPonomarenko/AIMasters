#include <iostream>
#include <vector>
#include <queue>

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);

    size_t n, m;
    std::cin >> n >> m;

    std::vector<uint32_t> v(n, 0);

    for (int i = 0;i < m;++i) {
        uint32_t a, b;
        std::cin >> a >> b;
        v[--a] += 1;
        b--;
        if (b < n - 1) {
            v[b + 1] -= 1;
        }
    }

    uint32_t k = 0;
    for (auto i : v) {
        std::cout << (k += i) << " ";
    }
}