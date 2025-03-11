#include <iostream>
#include <string>

int main()
{
    int l = 1, r, ans, m;
    std::cin >> r;

    do
    {
        m = (r + l + 1) / 2;
        std::cout << m << "\n";
        std::cout.flush();

        std::cin >> ans;
        if (ans == 0) {
            r = m - 1;
        } else if (ans == 2) {
            l = m + 1;
        }
    } while (ans != 1);
}