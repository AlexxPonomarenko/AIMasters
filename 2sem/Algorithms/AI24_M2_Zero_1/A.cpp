#include <iostream>
#include <vector>
#include <set>
#include <deque>

struct Graph
{
    size_t vertex_num = 0;
    uint32_t root;

    std::vector<std::vector<uint32_t>> e;
    Graph(uint32_t n) {
        for (uint32_t i = 0;i < n;++i) e.push_back(std::vector<uint32_t>());
        vertex_num = n;
    }
    void add_edge(uint32_t f, uint32_t t)
    {
        if (t == 0) {
            root = f;
        } else {
            e[f].push_back(t - 1);
            e[t - 1].push_back(f);
        }
    }
    const std::vector<uint32_t>& sosedi(uint32_t t) { return e[t]; }
    
};

struct DFS
{
    std::vector<uint32_t> d, f;
    Graph *G;
    DFS(Graph *G) { 
        uint32_t n = G->vertex_num;
        for (uint32_t i = 0;i < n;++i) {
            f.push_back(0);
            d.push_back(0);
        }
        this->G = G;
    }
    void operator()() {
        std::vector<std::pair<uint32_t, uint32_t>> s;
        uint32_t t = 0;

        s.emplace_back(G->root, 0);
        while (s.size()) {
            auto [v, i] = s.back(); s.pop_back();
            if (d[v] == 0) {
                d[v] = ++t;
            }
            auto &neigh = G->sosedi(v);
            for (; i < neigh.size();++i) {
                if (d[neigh[i]] == 0) {
                    s.emplace_back(v, i + 1);
                    s.emplace_back(neigh[i], 0);
                    break;
                }
            }
            if (i == neigh.size()) {
                f[v] = ++t;
            }
        }

        for (uint32_t u = 0;u < G->vertex_num;++u) {
            if (d[u]) continue;

            s.emplace_back(u, 0);
            while (s.size()) {
                auto [v, i] = s.back(); s.pop_back();
                if (d[v] == 0) {
                    d[v] = ++t;
                }
                auto &neigh = G->sosedi(v);
                for (; i < neigh.size();++i) {
                    if (d[neigh[i]] == 0) {
                        s.emplace_back(v, i + 1);
                        s.emplace_back(neigh[i], 0);
                        break;
                    }
                }
                if (i == neigh.size()) {
                    f[v] = ++t;
                }
            }
        }
    }
};

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);
    
    uint32_t n, m, t, t1, t2;
    std::cin >> n;

    Graph G(n);
    DFS dfs(&G);

    for (uint32_t i = 0;i < n;++i)
    {
        std::cin >> t;
        G.add_edge(i, t);
    }
    dfs();
    
    const auto &d = dfs.d, &f = dfs.f; 
    std::cin >> m;
    for (uint32_t i = 0;i < m;++i) {
        std::cin >> t1 >> t2;
        std::cout << ((d[t1 - 1] < d[t2 - 1]) && (f[t2 - 1] < f[t1 - 1])) << "\n"; std::cout.flush();
    }
}