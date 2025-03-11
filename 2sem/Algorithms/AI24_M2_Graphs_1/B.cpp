#include <iostream>
#include <vector>

struct DFS
{
private:
    void dfs(uint32_t cur_node, uint32_t parent_node, bool color) {
        colors_cnt[color] += 1;
        for (uint32_t i = 0;i < (*v)[cur_node].size();i++) {
            uint32_t p = (*v)[cur_node][i];
            if (parent_node != p) {
                dfs(p, cur_node, !color);
            }
        }
    }
    std::vector<std::vector<uint32_t>> *v;
    uint64_t colors_cnt[2] = {};
public:
    DFS(std::vector<std::vector<uint32_t>> *_v) : v(_v) {}

    uint64_t operator()() {
        std::vector<bool> already_was(v->size(), false);
        dfs(0, 0, 0);
        return colors_cnt[0] * colors_cnt[1] - v->size() + 1;
    }
};


int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);

    uint32_t n, t1, t2;
    std::cin >> n;
    std::vector<std::vector<uint32_t>> v(n, std::vector<uint32_t>());

    for (uint32_t i = 0;i < n - 1;++i)
    {
        std::cin >> t1 >> t2;
        v[t1 - 1].push_back(t2 - 1);
        v[t2 - 1].push_back(t1 - 1);
    }

    DFS dfs(&v);
    std::cout << dfs() << std::endl;
}