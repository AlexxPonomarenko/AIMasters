#include <iostream>
#include <vector>

#define INF ((1000000000 + 1))

class SegmentTree
{
    std::vector<int64_t> v;
    uint32_t h;
    uint32_t n_num;

    uint32_t l_child(uint32_t i) { return 2*i+1; }
    uint32_t r_child(uint32_t i) { return 2*i+2; }
    uint32_t parent(uint32_t i) { return i / 2; }

    void __build(uint32_t i)
    {
        if (l_child(i) < v.size()) {
            __build(l_child(i));
            v[i] += v[l_child(i)];
        }
        if (r_child(i) < v.size()) {
            __build(r_child(i));
            v[i] += v[r_child(i)];
        }
        return;
    }
    int64_t __func(uint32_t me, uint32_t wl, uint32_t wr, uint32_t l, uint32_t r) {
        
        if (l > r) return 0;
	    if (l == wl && r == wr)
		    return v[me];

        uint32_t mid = (wl + wr) / 2;

        return (
            __func(l_child(me), wl, mid, l, std::min(mid, r)) + 
            __func(r_child(me), mid + 1, r, std::max(mid + 1, l), wr)
        );
    }
    
public:
    SegmentTree(std::vector<int64_t> &a) 
    { 
        uint32_t n = a.size();
        uint32_t t = 1 << 31;

        while (!(n & t)) {
            t >>= 1;
            h--;
        }
        t <<= 1; 
        n_num = t;

        for (int i = 0;i < 2*t - 1;i++) v.push_back(0);

        for (int i = 0;i < n;++i) 
            v[t - 1 + i] = a[i];

        __build(0);
    }
    void print() { for (auto x : this->v) std::cout << x << " "; std::cout << "\n"; }
    int64_t func(uint32_t l, uint32_t r) {
        return __func(0, 0, n_num - 1, l - 1, r - 1);
    }

};

int main()
{
    uint32_t n, m;
    std::cin >> n;

    std::vector<int64_t> v;

    for (int i = 0;i < n;++i)
    {
        int64_t t;
        std::cin >> t; 
        v.push_back(t);
    }

    SegmentTree st(v);

    std::cin >> m;
    for (int i = 0;i < m;++i)
    {
        uint32_t l, r;
        std::cin >> l >> r;
        
        std::cout << st.func(l, r) << "\n"; std::cout.flush();
    }
}