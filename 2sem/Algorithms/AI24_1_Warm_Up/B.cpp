#include <iostream>
#include <vector>
#include <algorithm>

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);

    uint32_t n = 0, t = 0;
    uint64_t m1 = 0, m3 = 0, m5 = 0, m15 = 0, m15_2 = 0;
    
    std::cin >> n;
    for (int i = 0;i < n;++i)
    {
        std::cin >> t;
        bool div3 = !(t % 3), div5 = !(t % 5);

        if (!div3 && !div5 && t > m1) {
            m1 = t;
        } else if (div3 && !div5 && t > m3) {
            m3 = t;
        } else if (!div3 && div5 && t > m5) {
            m5 = t;
        } else if (div3 && div5 && t > m15) {
            m15_2 = m15;
            m15 = t;
        } else if (div3 && div5 && t > m15_2) {
            m15_2 = t;
        } 
    }

    std::vector<uint64_t> v = {m1 * m15, m3 * m5, m15 * m15_2, m15 * m3, m15 * m5};
    std::sort(v.begin(), v.end());
    std::cout << v[4] << std::endl;
}