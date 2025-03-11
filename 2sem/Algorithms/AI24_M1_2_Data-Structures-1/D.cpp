#include <iostream>
#include <string>
#include <algorithm>
#include <stack>

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);

    std::string s;
    std::string tmp_str;
    std::string t;

    std::cin >> s;

    std::stack<char> ops;
    std::vector<std::string> pol;

    bool wast = false;

    for (int i = 0;i < s.size();++i)
    {
        char c = s[i];
        switch (c)
        {
        case '(':
            ops.push(c);
            break;
        case ')':
            if (!tmp_str.empty()) {
                pol.push_back(tmp_str);
                tmp_str = "";
            }
            while(ops.top() != '(') {
                ops.pop();
            }
            ops.pop();
            break;
        case 'R':
            pol.push(std::string{c});
            break;
        case ',':
            break;
        default:
            tmp_str += c;
            break;
        }
    }

    while (!pol.empty())
    {
        tmp_str = pol.top(); pol.pop();
        if (!pol.empty())
    }

}