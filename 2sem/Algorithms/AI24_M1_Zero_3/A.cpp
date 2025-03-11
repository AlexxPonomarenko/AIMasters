#include <iostream>
#include <map>

int main()
{
    std::map<uint64_t, uint64_t> line;

    size_t n;
    std::cin >> n;
    
    for (size_t i = 0; i < n; i++)
    {
        uint64_t p1, p2;
        std::cin >> p1 >> p2;
        
        if (line.find(p1) == line.end()) {
            line[p1] = p1;
        }
        if (line.find(p2) == line.end()) {
            line[p2] = p2;
        }

        std::swap(line[p1], line[p2]);
        if (line[p1] >= line[p2]) std::cout << line[p1] - line[p2] << "\n"; 
        if (line[p2] > line[p1]) std::cout << line[p2] - line[p1] << "\n"; 
        std::cout.flush();

    }
    
}