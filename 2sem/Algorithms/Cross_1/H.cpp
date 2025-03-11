#include <iostream>
#include <stack>
#include <vector>
#include <string>

#define MOD (1000000000 + 7)

struct State
{
    int64_t num;
    char op;
    State(int64_t num = 0, char op = 'n') : num(num), op(op) {}
    bool operator!=(char c) { return op != c; }
    int64_t operator*(const State &s) const { return ((num * s.num) % MOD + MOD) % MOD; }
    int64_t operator-(const State &s) const { return ((num - s.num) % MOD + MOD) % MOD; }
    int64_t operator+(const State &s) const { return ((num + s.num) % MOD + MOD) % MOD; }
};

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);
    uint32_t n;
    std::cin >> n;
    
    std::vector<State> rpn;
    std::stack<State> ops;
    
    std::string c;
    for (int i = 0;i < n;++i)
    {
        std::cin >> c;
        if (std::isdigit(c[0])) { 
            rpn.push_back(State(std::stoi(c)));
        } 
        else {
            switch (c[0])
            {
            case ')':
                while (ops.top() != '(')
                {
                    rpn.push_back(ops.top()); ops.pop();                
                }
                ops.pop();
                break;
            case '-':
            case '+':
                while (ops.size() && ops.top().op != '(') 
                {
                    rpn.push_back(ops.top()); ops.pop();
                }
            case '(':
            case '*':
                ops.push(State(0, c[0]));
                break;
            }
        }
    }
    while (ops.size()) {
        rpn.push_back(ops.top()); ops.pop();
    }

    for (int i = 0;i < rpn.size();++i)
    {
        State cur, prev;
        switch (rpn[i].op)
        {
        case 'n':
            ops.push(rpn[i]);
            break;
        default:
            cur = ops.top(); ops.pop();
            prev = ops.top(); ops.pop();
            switch (rpn[i].op)
            {
            case '+':
                ops.push(prev + cur); break;
            case '-':
                ops.push(prev - cur); break;
            case '*':
                ops.push(prev * cur); break;
            }
            break;
        }
    }
    std::cout << (ops.top().num % MOD + MOD) % MOD << std::endl;
}