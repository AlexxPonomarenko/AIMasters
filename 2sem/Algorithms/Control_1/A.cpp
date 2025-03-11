#include <iostream>
#include <vector>
#include <map>
#include <deque>

void solve()
{
    uint32_t n, m;
    std::cin >> n >> m;

    std::vector<int64_t> post_del(n, -1);
    uint32_t p = 0;
    std::map<uint32_t, uint32_t> added;

    uint32_t t;
    for (int i = 0;i < m;++i)
    {
        std::cin >> t;
        if (added.find(t) == added.end())
        {
            added[t] = 1;
            if (p < n)
                post_del[p++] = i + 1;
        }
    }
    for (int i = n - 1;i >= 0;i--)
    {
        std::cout << post_del[i] << " ";
    }
    std::cout << "\n";
}

int main()
{
    // std::ios::sync_with_stdio(false); std::cin.tie(0);

    uint32_t t;
    std::cin >> t;
    for (int i = 0;i < t;i++)
    {
        solve();
    }    
}
