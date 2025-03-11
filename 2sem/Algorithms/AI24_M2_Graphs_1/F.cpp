#include <iostream>
#include <vector>

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);

    uint32_t n, t;
    std::cin >> n >> t;

    std::vector<uint32_t> v(n + 1);

    for (uint32_t i = 1;i < n;++i)
    {
        std::cin >> v[i];
    }
    v[n] = 1;

    uint32_t cur = 1;
    for (; cur <= n; cur += v[cur]) {
        if (cur == t) break;
    }

    if (cur == t)    
        std::cout << "YES" << std::endl;
    else
        std::cout << "NO" << std::endl;
}