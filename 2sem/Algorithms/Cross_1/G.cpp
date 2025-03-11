#include <iostream>
#include <queue>
#include <map>
#include <vector>
#include <string>

struct Query
{
    uint32_t q;
    uint32_t num;
    uint32_t status;

    Query(int32_t _q, uint32_t _num) : q(_q), num(_num), status(1) {}
    bool operator<(const Query &s) const 
    { 
        return q < s.q; 
    }
};


int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);
    uint32_t n, m;
    uint64_t total = 0, cur_q = 0, money;
    char query;

    std::cin >> n >> m;
    std::vector<uint32_t> queries(n + m, 1);
    std::priority_queue<Query> q;

    for (int i = 0;i < n + m;++i)
    {
        std::cin >> query >> money;
        Query t = Query(money, i);

        if (query == '+') {
            total += t.q;
            queries[i] = 2;
        }
        else {
            if (cur_q + t.q <= total) {
                q.push(t);
                cur_q += t.q;
            } else {
                if (!q.empty() && t.q < q.top().q) {
                    cur_q = cur_q - q.top().q + t.q;
                    queries[q.top().num] = 0;
                    q.pop();
                    q.push(t);
                } else {
                    queries[i] = 0;
                }
            }
        }
    }

    std::vector<std::string> print = {"declined", "approved", "resupplied"};
    for (const auto &s : queries)
    {
        std::cout << print[s] << std::endl;
    }

    return 0;
}