#include <iostream>
#include <vector>

uint64_t free_space(const std::vector<uint64_t> &v, uint64_t h)
{
    uint64_t s = 0;
    for (uint64_t height : v)
        s += height >= h ? 0 : h - height;
    return s;
}

void solve()
{
    uint64_t n, x, s = 0;
    std::cin >> n >> x;
    std::vector<uint64_t> v(n, 0);
    
    for (int i = 0;i < n;++i) {
        std::cin >> v[i];
        // if (v[i] > s) s = v[i];
        s += v[i];
    }
    
    uint64_t l = 1, r = (s + x) / n + n, m, h;
    while (l <= r)
    {
        m = (l + r) / 2;
        s = free_space(v, m);
        if (s <= x) {
            l = m + 1;
            h = m;
        } else {
            r = m - 1;
        }
    }
    std::cout << h << "\n"; std::cout.flush();
}

int main()
{
    uint64_t t;
    std::cin >> t;
    for (int i = 0;i < t;++i)
        solve();
}