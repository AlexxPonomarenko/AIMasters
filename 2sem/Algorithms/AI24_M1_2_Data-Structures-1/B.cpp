#include <iostream>
#include <string>
#include <queue>

uint32_t convert(std::string &s)
{
    int k = 0;
    int l = 0;
    if (s[2] == ':')
    {
        k = (s[0] - '0') * 10 + (s[1] - '0');
        l = 3;
    } else {
        k = (s[0] - '0');
        l = 2;
    }
    k = k * 60 + (s[l] - '0') * 10 + (s[l + 1] - '0');
    return k;
}

struct Man
{
    uint32_t start, end;
    std::string name;
    Man(uint32_t _start, uint32_t _end, std::string &s)
    {
        start = _start, end = _end;
        name.reserve(10);
        name = s;
    }
    // cucha max
    bool operator<(Man const &m) const
    {
        return end > m.end;
    }
    void print() const { std::cout << name << " " << start << " " << end << "\n"; std::cout.flush(); }
};

int main()
{
    uint32_t n, m;
    std::cin >> n >> m;

    std::priority_queue<Man> pq;

    std::string name, s_start, s_end;
    name.reserve(10), s_start.reserve(5), s_end.reserve(5);
    uint32_t start, end;
    
    for (int i = 0;i < n;++i)
    {
        std::cin >> name >> s_start >> s_end;
        start = convert(s_start), end = convert(s_end);
        Man chel(start, end, name);

        // release 
        while (!pq.empty() && start > pq.top().end)
        {
            pq.pop();
        }        

        if (pq.size() >= m) {
            std::cout << "No\n"; std::cout.flush();
        } else {
            pq.push(chel);
            std::cout << chel.name << "\n"; std::cout.flush();
        }
    }
}