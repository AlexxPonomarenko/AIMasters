#include <iostream>
#include <vector>

int main()
{
    int n;
    std::cin >> n;

    std::vector<size_t> d;
    d.reserve(n);

    std::vector<int64_t> a;
    a.reserve(n);

    std::cin >> a[0];
    d[0] = 1;
    std::cout << d[0] << "\n"; std::cout.flush();

    int64_t cur_max = 1;
    for (int i = 1;i < n;++i)
    {   
        std::cin >> a[i];
        int64_t max = 0;
        for (int j = 0; j < i;j++)
        {
            if (a[i] >= a[j] && d[j]> max)
            {
                max = d[j];
            }
        }
        d[i] = max + 1;
        if (d[i] > cur_max) cur_max = d[i];
        std::cout << cur_max << "\n"; std::cout.flush();
    }

    for (int i = 0;i < n;++i)
    {
        std::cout << d[i] << " ";
    }
}