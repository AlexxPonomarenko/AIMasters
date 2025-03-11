#include <iostream>
#include <algorithm>
#include <vector>

int main()
{
    std::uint32_t t, n, s;
    std::vector<uint32_t> v(100, 0);
    v.reserve(100);

    std::cin >> t;
    for (int i = 0;i < t;++i)
    {
        std::cin >> n;
        for (int j = 0;j < n;++j)
        {
            std::cin >> v[j];
        }
        
        std::sort(v.begin(), v.begin() + n);
        
        s = 0;
        for (int j = n / 2 + n % 2;j < n;++j)
            s += v[j];
        for (int j = 0;j < n / 2;++j)
            s -= v[j];
        
        std::cout << s << "\n";
    }
}