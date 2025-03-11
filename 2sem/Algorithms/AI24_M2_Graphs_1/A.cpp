#include <iostream>
#include <vector>
#include <set>

struct DFS
{
    std::vector<uint32_t> d;
    std::vector<std::vector<uint32_t>> *v;
    std::vector<uint32_t> component;
    std::vector<uint32_t> comp_num;
    uint32_t num_component = 0;


    DFS(std::vector<std::vector<uint32_t>> *_v) : v(_v) {
        for (uint32_t i = 0;i < v->size();++i) {
            component.push_back(-1);
            d.push_back(0);
        }
    }

    uint32_t dfs(uint32_t cur_node) {
        uint32_t s = 1;
        for (uint32_t sosed_num : (*v)[cur_node]) {
            if (d[sosed_num])
                continue;
            d[sosed_num] = 1;
            
            component[sosed_num] = num_component;
            s += dfs(sosed_num);
        }
        return s;
    }

    void operator()() {
        for (uint32_t i = 0;i < v->size();++i) {
            if (component[i] != uint32_t(-1)) continue;

            comp_num.push_back(dfs(i));
            num_component++;
        }
    }   
};


int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);

    uint32_t n, m, k, t;
    std::cin >> n >> m;
    std::vector<std::vector<uint32_t>> v(n, std::vector<uint32_t>());
    std::vector<std::vector<uint32_t>> users(m, std::vector<uint32_t>());
    std::vector<uint32_t> cur_cnt(n, 0);

    for (uint32_t i = 0;i < m;++i) {
        std::cin >> k;
        for (uint32_t j = 0;j < k;++j) {
            std::cin >> t;
            users[i].push_back(t - 1);
        }
    }

    for (uint32_t i = 0; i < m;++i) {
        for (uint32_t j = 1; j < users[i].size();j++) {
            v[users[i][0]].push_back(users[i][j]);
            v[users[i][j]].push_back(users[i][0]);
        }
    }

    DFS dfs(&v);
    dfs();

    for (uint32_t i = 0;i < n;++i) {
        if (dfs.component[i] == uint32_t(-1)) {
            std::cout << 1 << " ";
            continue;
        }
        uint32_t f = dfs.comp_num[dfs.component[i]];
        std::cout << (f == 1 ? f : f - 1) << " ";
    }
    std::cout << std::endl;
}