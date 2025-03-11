#include <iostream>
#include <string>
#include <vector>

std::vector<uint32_t> prefix_func(const std::string &s, uint32_t len_b, std::vector<uint32_t> &pref)
{
    uint32_t k = 0;
    std::vector<uint32_t> pos;
    for (uint32_t i = 1;i < s.size();++i) {
        k = pref[i - 1];
        while (k > 0 && s[i] != s[k])
            k = pref[k - 1];
        if (s[i] == s[k])
            k++;
        pref[i] = k;
        if (pref[i] == len_b) {
            pos.push_back(i - 2 * len_b);
        } 
    }
    return pos;
}

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);
    std::string s_a, s_b;
    std::cin >> s_a >> s_b;

    std::vector<uint32_t> pref(s_a.size() + s_b.size() + 1, 0);
    auto pos = prefix_func(s_b + "@" + s_a, s_b.size(), pref);

    std::cout << pos.size() << "\n";
    for (uint32_t i = 0;i < pos.size();++i) {
        std::cout << pos[i] + 1 << " ";
    }

    return 0;
}