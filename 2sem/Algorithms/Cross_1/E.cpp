#include <iostream>
#include <vector>

void solve(uint32_t n)
{
    std::vector<uint32_t> p(n, 0);
    std::vector<bool> was_before(n, false);
    uint32_t cnt = 0;

    for (int i = 0;i < n;i++)
    {
        std::cin >> p[i];
        was_before[p[i]] = true;
        if (p[i] > 1 && !was_before[p[i] - 1])
            cnt += 1;
    }
    std::cout << cnt << "\n"; std::cout.flush();    
}

int main()
{
    uint32_t t, n;
    std::cin >> t;
    for (int i = 0;i < t;++i)
    {
        std::cin >> n;
        solve(n);
    }
}