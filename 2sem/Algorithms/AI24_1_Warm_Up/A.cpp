#include <iostream>
#include <limits>

int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);
    float eps = std::numeric_limits<float>::epsilon();

    float x1_r, x2_r, y3_r = 0;
    float x1_l, x2_l, y3_l = 0;
    x1_r = 1000 + eps, x2_r = 0;
    x1_l = 0, x2_l = -1000 - eps;

    bool f1_r = false, f2_r = false, f3_r = false;
    bool f1_l = false, f2_l = false, f3_l = false;

    int n;
    std::cin >> n;
    for (int i = 0;i < n;++i)
    {
        float x, y;
        std::cin >> x >> y;
        y = std::abs(y);

        if (x < 0) {
            if (y < eps) {
                if (x < x1_l) {
                    x1_l = x;
                    f1_l = true;
                } else if (x > x2_l) {
                    x2_l = x;
                    f2_l = true;
                }
            } else if (y > y3_l) {
                y3_l = y;
                f3_l = true;
            }
        } else if (x > 0) {
            if (y < eps) {
                if (x < x1_r) {
                    x1_r = x;
                    f1_r = true;
                } else if (x > x2_r) {
                    x2_r = x;
                    f2_r = true;
                }
            } else if (y > y3_r) {
                y3_r = y;
                f3_r = true;
            }
        }
        

    }
    std::cout << std::max(
            (f1_l * f2_l * f3_l) * (x2_l - x1_l) * y3_l / 2,
            (f1_r * f2_r * f3_r) * (x2_r - x1_r) * y3_r / 2
        ) << std::endl;
}