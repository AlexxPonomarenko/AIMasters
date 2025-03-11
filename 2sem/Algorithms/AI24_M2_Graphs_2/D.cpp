#include <iostream>
#include <vector>

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
        e[f].push_back(t);
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
        for 
    }
};

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);
    uint32_t n, m, t, t1, t2;
    std::cin >> n;

    Graph G(n);

    for (uint32_t i = 0;i < n;++i)
    {
        for (uint32_t j = 0;j < n;++j) {
            std::cin >> t;
            if (t) {
                G.add_edge(i, j);
                G.add_edge(j, i);
            }
        }
    }

    DFS dfs(&G);
    dfs();

}