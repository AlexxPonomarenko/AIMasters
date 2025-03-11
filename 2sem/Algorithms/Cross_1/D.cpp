#include <iostream>
#include <string>

int main()
{
    uint32_t n, m;
    std::cin >> n >> m;
    uint32_t cnt_pairs = 0;

    std::string answ;
    bool is_lower = false;
    for (uint32_t i = 1;i <= n;++i)
    {
        uint32_t p_cur = 0;
        while (1)
        {
            std::cout << "? " << i << " " << p_cur + 1 << "\n"; std::cout.flush();
            std::cin >> answ;
            if (answ == "NO") 
            {
                break;
            }
            is_lower = true;
            p_cur++;
            if (p_cur >= m) break;
        }
        cnt_pairs += m - p_cur; 
    }
    std::cout << "! " << cnt_pairs << "\n"; std::cout.flush();
}