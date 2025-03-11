#include <iostream>
#include <vector>

struct SegNode
{
    uint64_t val;
    size_t lb, rb, index;
    SegNode *left = nullptr, *right = nullptr;
    

    SegNode(uint64_t _val, size_t _lb, size_t _rb, size_t _index) : val(_val), lb(_lb), rb(_rb), index(_index) {}
    static size_t left_child(size_t index) { return index * 2 + 1; }
    static size_t right_child(size_t index) { return index * 2 + 2; }
};

class SegTree
{
    SegNode *root;
    size_t total_size;
    void _create(const std::vector<uint64_t> &v, SegNode* cur, size_t index) {
        size_t l = SegNode::left_child(index), r = SegNode::right_child(index);
        size_t lb = cur->lb, rb = cur->rb;
        size_t mid = (lb + rb) / 2;

        if (l < total_size - 1) {
            cur->left = new SegNode(v[l], lb, mid, l);
            _create(v, cur->left, l);
        }
        if (r < total_size) {
            cur->right = new SegNode(v[r], mid, rb, r);
            _create(v, cur->right, r);
        }
    }
    uint64_t _sum(SegNode *cur, size_t l, size_t r) {
        size_t wl = cur->lb, wr = cur->rb; 
        
        if (wl >= r || wr <= l) return 0;
        if (wl >= l && wr <= r) return cur->val;

        uint32_t mid = (wl + wr) / 2;

        return _sum(cur->left, l, r) + _sum(cur->right, l, r);
    }

public:
    SegTree(const std::vector<uint64_t> &_v) {
        total_size = _v.size();
        root = new SegNode(_v[0], 0, (total_size + 1) / 2, 0);
        _create(_v, root, 0);
    }

    uint64_t sum(size_t l, size_t r) { return _sum(root, l, r); }
    void add(size_t l, size_t r, uint64_t val) {}
    
};


int main()
{
    uint32_t n, m;
    std::cin >> n;

    uint32_t t = 1 << 31;
    while (!(n & t)) {
        t >>= 1;
    }
    t <<= 1;

    std::vector<uint64_t> v(2*t - 1, 0);
    for (int i = 0;i < n;++i)
    {
        uint64_t tmp;
        std::cin >> tmp; 
        v[t + i - 1] = tmp;
    }
    for (int i = t - 2; i >= 0; --i) {
        v[i] = v[i * 2 + 1] + v[i * 2 + 2];
    }

    SegTree st(v);

    std::cin >> m;
    for (int i = 0;i < m;++i)
    {
        uint32_t l, r, val;
        char q;
        std::cin >> q;
        std::cin >> l >> r;

        if (q == 'Q') {
            std::cout << st.sum(l - 1, r) << "\n"; 
            std::cout.flush();
        } else {
            std::cin >> val;
            st.add(l, r, val);
        }
    }
}