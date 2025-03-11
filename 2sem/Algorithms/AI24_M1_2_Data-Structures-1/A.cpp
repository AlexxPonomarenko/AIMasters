#include <iostream>
#include <vector>
#include <queue>

struct Seg
{
    uint32_t left, right;
    Seg(uint32_t _left = 0, uint32_t _right = 0) : left(_left), right(_right) {}
    bool operator<(Seg const &m) const { 
        if (left == m.left) {
            return right > m.right;
        }
        return left > m.left; }
    void print() const { std::cout << "(" << left + 1 << ":" << right + 1 << ")"; }
};

void print(uint32_t l, uint32_t r, uint32_t n)
{
    for (int i = l;i < r;i++) {
        std::cout << n << " "; std::cout.flush();
    }
}

int main()
{
    // std::ios::sync_with_stdio(false); std::cin.tie(0);

    size_t n, m;
    std::cin >> n >> m;

    std::priority_queue<Seg> pq;

    // on top is leftest segment
    for (int i = 0;i < m;++i) {
        uint32_t a, b;
        std::cin >> a >> b;
        a--, b--;
        pq.push(Seg(a, b));
    }

    uint32_t k = 0;
    print(0, pq.top().left, k++);
    uint32_t l = 0, r = 0;
    while (!pq.empty())
    {
        Seg cur = pq.top(); pq.pop();
        uint32_t l_next = pq.top().left;
        

        if (cur.right < l_next) {
            print(cur.left, cur.right, k);
            print(cur.right, l_next, k--);
        } else {
            print(cur.left, l_next, k);
            k++;
        }
    }
}