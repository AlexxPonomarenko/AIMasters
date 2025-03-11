#include <iostream>
#include <vector>

int64_t sum(const std::vector<int64_t> &v, uint32_t i, uint32_t l, uint32_t r, uint32_t wl, uint32_t wr) {
  if (wl >= r || wr <= l) return 0;     // Outside of range
  if (wl >= l && wr <= r) return v[i];  // Inside of range
  
  uint32_t mid = (wl + wr) / 2;

  return sum(v, i * 2 + 1, l, r, wl, mid) + sum(v, i * 2 + 2, l, r, mid, wr);
}


int main()
{
    uint32_t n, m;
    std::cin >> n;

    uint32_t t = 1 << 31;
    while (!(n & t)) {
        t >>= 1;
    }
    t <<= 1;

    std::vector<int64_t> v(2*t - 1, 0);
    for (int i = 0;i < n;++i)
    {
        int64_t tmp;
        std::cin >> tmp; 
        v[t + i] = tmp;
    }

    // build
    for (int i = t - 1; i >= 0; --i) {
        v[i] = v[i * 2 + 1] + v[i * 2 + 2];
    }

    std::cin >> m;
    for (int i = 0;i < m;++i)
    {
        uint32_t l, r;
        std::cin >> l >> r;
        
        std::cout << sum(v, 0, l--, r + 1, 0, t) << "\n"; std::cout.flush();
    }
}