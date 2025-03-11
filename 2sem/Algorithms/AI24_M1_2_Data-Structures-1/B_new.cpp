#include <iostream>
#include <queue>
#include <cstring>

uint32_t convert(const char* s)
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
    char name[11];
    Man(uint32_t _start, uint32_t _end, const char *s)
    {
        start = _start, end = _end;
        strncpy(name, s, 11);
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
    scanf("%u %u", &n, &m);

    std::priority_queue<Man> pq;

    char name[11], s_start[6], s_end[6];
    uint32_t start, end;

    for (int i = 0;i < n;++i)
    {
        scanf("%s %s %s", name, s_start, s_end);
        start = convert(s_start), end = convert(s_end);
        Man chel(start, end, name);

        // release 
        while (!pq.empty() && start > pq.top().end)
        {
            pq.pop();
        }        

        if (pq.size() >= m) {
            printf("No\n"); fflush(stdout);
        } else {
            pq.push(chel);
            printf("%s\n", chel.name); fflush(stdout);
        }
    }
}