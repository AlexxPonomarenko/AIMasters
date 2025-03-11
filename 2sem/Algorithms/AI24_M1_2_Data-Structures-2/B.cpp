#include <iostream>
#include <vector>

struct SegNode
{
    uint64_t val, assign;
    size_t index, lb, rb;
    bool assign_flag = false;
    SegNode *left = nullptr, *right = nullptr;

    SegNode(uint64_t _val, size_t _lb, size_t _rb, size_t _index) : val(_val), lb(_lb), rb(_rb), index(_index) {}
    static size_t left_child(size_t index) { return index * 2 + 1; }
    static size_t right_child(size_t index) { return index * 2 + 2; }
    void update(size_t l, size_t r)
    {
        if (assign_flag) {
            if (left) {
                left->set_assign(assign);
                left->val += assign * (std::min(left->rb, r) - std::max(left->lb, l));
                right->set_assign(assign);
                right->val += assign * (std::min(right->rb, r) - std::max(right->lb, l));
            }
        }
        assign_flag = false;
    }
    void set_assign(uint64_t _val) {
        assign = _val;
        assign_flag = true;
    }
};

class SegTree
{
    SegNode *root;
    size_t total_size;
    void _create(std::vector<uint64_t> &v, SegNode* cur, size_t index) {
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
        if (!cur) return 0;
        cur->update(l, r);
        size_t wl = cur->lb, wr = cur->rb; 
        
        if (wl >= r || wr <= l) return 0;
        if (wl <= l && wr <= r) return cur->val;

        uint32_t mid = (wl + wr) / 2;
        cur->val = _sum(cur->left, l, r) + _sum(cur->right, l, r);
        return cur->val;
    }
    void _add(SegNode *cur, size_t l, size_t r, uint64_t x) {
        size_t lb = cur->lb, rb = cur->rb;

        if (lb <= l && r <= rb) {
            cur->set_assign(x);
            cur->val += (r - l) * x;
        }
        else if (cur->left && std::max(lb, l) < std::min(r, rb)) {
            _add(cur->left, l, r, x);
            _add(cur->right, l, r, x);
        }
        cur->update(l, r);
    }
public:
    SegTree(std::vector<uint64_t> &_v) {
        total_size = _v.size();
        root = new SegNode(_v[0], 0, (total_size + 1) / 2, 0);
        _create(_v, root, 0);
    }

    uint64_t sum(size_t l, size_t r) { return _sum(root, l, r); }
    void add(size_t l, size_t r, uint64_t x) { return _add(root, l, r, x); }
    
};


int main()
{
    uint32_t n, m;
    std::cin >> n;

    uint32_t t = 1 << 31;
    while (!((n - 1) & t)) {
        t >>= 1;
    }
    t <<= 1;

    std::vector<uint64_t> v(2*t - 1, 0);
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
            st.add(l - 1, r, val);
        }
    }
}