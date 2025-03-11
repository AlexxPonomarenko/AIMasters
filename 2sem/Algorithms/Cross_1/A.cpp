#include <iostream>
#include <algorithm>
#include <string>
#include <map>
#include <vector>

struct elem
{
    std::string name;
    std::vector<uint32_t> values;    
    static std::vector<uint32_t> *priors;

    elem(const std::string &s) : name(s) {}
    bool operator<(const elem &s) const 
    {
        for (int i = 0;i < (*priors).size();++i) {
            uint32_t p = (*priors)[i];
            if (values[p] != s.values[p])
                return values[p] > s.values[p];
        }
        return false;
    }
};

std::vector<uint32_t>* elem::priors = nullptr;

int main()
{
    // std::ios::sync_with_stdio(false); std::cin.tie(0);
    uint32_t n, k, t;
    std::cin >> n >> k;
    
    std::vector<uint32_t> p(k, 0);
    for (int i = 0;i < k;++i)
    {
        std::cin >> t;
        p[t - 1] = i;
    } 
    elem::priors = &p;

    std::vector<elem> elems;
    for (int i = 0;i < n;++i)
    {
        std::string s;
        std::cin >> s;

        std::vector<uint32_t> vals(k, 0);
        for (int j = 0;j < k;++j)
        {
            std::cin >> vals[j];
        }
        elem t(s);
        t.values = vals;
        elems.push_back(t);
    }

    std::stable_sort(elems.begin(), elems.end());

    for (const auto &e : elems)
    {
        std::cout << e.name << "\n"; std::cout.flush();
    }
}