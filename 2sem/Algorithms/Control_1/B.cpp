#include <iostream>
#include <string>

uint32_t lookup_tree(bool is_in_black)
{
    char color, t;
    std::cin >> t >> color;
    if (color == 'b')
        is_in_black = true;

    uint32_t s = 0;
    while (1)
    {
        std::cin >> t;
        if (t == ')')
            break;

        if (t == ',')
        {
            s += lookup_tree(is_in_black);
        }
    }
    return s + is_in_black * (color == 'w');
}

int main()
{
    std::ios::sync_with_stdio(0); std::cin.tie(0);
    std::cout << lookup_tree(false) << "\n"; std::cout.flush();
}