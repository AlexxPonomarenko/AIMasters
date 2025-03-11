#include <iostream>
#include <vector>
#include <limits>

#define MINUS_INF std::numeric_limits<int64_t>::min()

int64_t get(int64_t t) {
    return t == MINUS_INF ? MINUS_INF : t;
}

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);
    int64_t n;
    std::cin >> n;
    std::vector<int64_t> v(n + 5, 0);

    for (int64_t i = 0;i < 5;i++)
        v[i] = MINUS_INF;

    for (int64_t i = 5; i < n + 5;++i) {
        char c;
        std::cin >> c;
        switch (c)
        {
        case 'w':
            v[i] = MINUS_INF;
            break;
        case '"':
            v[i] = 1;
            break;
        case '.':
            break;
        }
    }

    for (int64_t i = 5;i < n + 5;++i) {
        if (v[i] == MINUS_INF)
            continue;

        if (i - 5 == 0) 
            continue;
        else if (i - 5 == 1) {
            int64_t l = get(v[i - 1]);
            if (l == MINUS_INF) 
                v[i] = MINUS_INF;
            else
                v[i] += l;
        }
        else if (i - 5 == 2) {
            int64_t l = get(v[i - 1]);
            if (l == MINUS_INF) 
                v[i] = MINUS_INF;
            else
                v[i] += l;
        }
        else if (i - 5 == 3) {
            int64_t l = std::max(get(v[i - 1]), get(v[i - 3]));
            if (l == MINUS_INF) 
                v[i] = MINUS_INF;
            else
                v[i] += l;
        }
        else if (i - 5 == 4) {
            int64_t l = std::max(get(v[i - 1]), get(v[i - 3]));
            if (l == MINUS_INF) 
                v[i] = MINUS_INF;
            else
                v[i] += l;
        } 
        else {
            int64_t l = std::max(get(v[i - 1]), std::max(get(v[i - 3]), get(v[i - 5])));
            if (l == MINUS_INF) 
                v[i] = MINUS_INF;
            else
                v[i] += l;
        } 
            
    }
    std::cout << (v[n + 4] >= 0 ? v[n + 4] : -1) << std::endl;
}