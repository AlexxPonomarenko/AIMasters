#include <iostream>

std::pair<int64_t, int64_t> euqlid(int64_t a, int64_t b, int64_t c)
{
    if (b == 0) {
        if (!(c % a)) return {c / a, 0};
        else return {-1, -1};
    }
    std::pair<int64_t, int64_t> p = euqlid(b, a % b, c % b);
    if (p.second == -1 && p.first == -1) return {-1, -1};
    return {p.second, (c - a * p.second) / b};
}

std::pair<int64_t, int64_t> fit_solve(std::pair<int64_t, int64_t> solve, int64_t a, int64_t b)
{
    int64_t x0 = solve.first, y0 = solve.second;
    // int64_t t = 0;

    // if (x0 <= 0)
    // {   
    //     if (b > 0)
    //         t = x0 / b - 1;
    //     else
    //         t = -x0 / b - 1;
    // }
    // else if (x0 > 0)
    // {
    //     if (b > 0)
    //         t = (x0 % b) ? x0 / b : x0 / b - 1;
    //     else
    //         t = (x0 % b) ? -x0 / b : -x0 / b - 1;

    // }
    // x0 = x0 - b * t;
    // y0 = y0 + a * t;

    while (x0 > 0) {
        if (b > 0) {
            x0 -= b;
            y0 += a;
        } else if (b < 0) {
            x0 += b;
            y0 -= a;
        } else {
            break;
        }
    }

    while (x0 <= 0) {
        if (b > 0) {
            x0 += b;
            y0 -= a;
        } else if (b < 0) {
            x0 -= b;
            y0 += a;
        } else {
            break;
        }
    }

    return {x0, y0};
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

    if (b == 0 && (c % a || c / a < 0)) {
        std::cout << "No\n";
        return 0;
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

    p = fit_solve(p, a, b);
    int64_t x0 = p.first, y0 = p.second;

    std::cout << x0 << " " << y0  << "\n";
}