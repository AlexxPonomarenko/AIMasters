#include <iostream>
#include <vector>

struct DFS
{
private:
    uint32_t dfs(uint32_t cur_node) {
        if ((*v)[cur_node].size() == 0) {
            return 1;
        }

        uint32_t cur_depth = 0;
        for (uint32_t i = 0;i < (*v)[cur_node].size();i++) {
            uint32_t p = (*v)[cur_node][i];
            cur_depth = std::max(dfs(p) + 1, cur_depth);
        }
        return cur_depth;
    }
    std::vector<std::vector<uint32_t>> *v;

public:
    std::vector<uint32_t> node_depth;
    DFS(std::vector<std::vector<uint32_t>> *_v) : v(_v) { for (uint32_t i=0;i<v->size();++i) node_depth.push_back(0); }

    uint32_t operator()() {
        uint32_t max_depth = 0;
        for (uint32_t i = 0;i < (*v).size();++i) {
            if (!node_depth[i]) {
                node_depth[i] = dfs(i); 
            }
            max_depth = std::max(max_depth, node_depth[i]);
        }
        return max_depth;
    }
};


int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);

    uint32_t n, t;
    std::cin >> n;
    std::vector<std::vector<uint32_t>> v(n, std::vector<uint32_t>());

    for (uint32_t i = 0;i < n;++i)
    {
        std::cin >> t;
        if (t != -1)
            v[t - 1].push_back(i);
    }

    DFS dfs(&v);
    std::cout << dfs() << std::endl;
}