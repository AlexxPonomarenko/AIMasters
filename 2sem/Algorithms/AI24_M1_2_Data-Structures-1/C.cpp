#include <iostream>
#include <queue>

struct quota
{
    uint64_t p_num;
    double q;
    quota (uint64_t _p, double _q) : p_num(_p), q(_q) {}
    bool operator<(const quota &a) const {
        if (q == a.q) return p_num > a.p_num;
        else return q < a.q;
    }
};

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);

    uint64_t n, m;
    std::cin >> n >> m;

    std::priority_queue<quota> pq;
    std::vector<uint64_t> part(n, 0);
    std::vector<uint64_t> voices(n, 0);

    for (int i = 0;i < n;++i) {
        uint64_t v;
        std::cin >> v;
        voices[i] = v;
        pq.push(quota(i, (double)v));
    }

    while (m--) {
        quota t = pq.top();
        part[t.p_num] += 1;

        pq.pop();
        pq.push(quota(t.p_num, voices[t.p_num] / ((double)part[t.p_num] + 1)));
    }

    for (auto x : part) std::cout << x << " ";
}