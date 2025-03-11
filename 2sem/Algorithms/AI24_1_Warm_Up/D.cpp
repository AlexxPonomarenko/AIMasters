#include <iostream>

std::pair<int64_t, int64_t> euqlid(int64_t a, int64_t b, int64_t c)
{
    int64_t x0 = 1, x1 = 0, y0 = 0, y1 = 1, r0 = a, r1 = b, q;
    int64_t t;

    while (r1 != 0)
    {
        t = r1;
        q = r0 / r1;
        r1 = r0 % r1;
        r0 = t;
        
        t = y1;
        y1 = y0 - q * y1;
        y0 = t;

        t = x1;
        x1 = x0 - q * x1;
        x0 = t;
    }
    if (c % r0) return {-1, -1};
    else return {x0 * c / r0, y0 * c / r0};
}

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);

    int64_t a, b, c;
    bool swch = false;
    std::cin >> a >> b >> c;
    
    if (a < b) {
        std::swap(a, b);
        swch = true;
    }

    auto p = euqlid(a, b, c);
    if (swch) 
    {
        std::swap(p.second, p.first);
        std::swap(a, b);
    }

    if (p.second == -1 && p.first == -1) {
        std::cout << "No\n";
        return 0;
    }

    int64_t x0 = p.first, y0 = p.second;
    
    while (x0 - b > 0)
    {
        x0 -= b;
        y0 += a;
    }

    while (x0 <= 0) {
        x0 += b;
        y0 -= a;
    }

    std::cout << x0 << " " << y0  << "\n";
}