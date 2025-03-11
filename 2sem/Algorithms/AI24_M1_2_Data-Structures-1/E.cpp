#include <iostream>
#include <queue>

int main()
{

    uint64_t t, n, k, sum;
    std::cin >> t;
    for (int i = 0;i < t;++i) 
    {
        std::cin >> n;
        sum = 0;
        std::priority_queue<uint64_t> pq;
        // while (!pq.empty()) pq.pop();

        for (int j = 0;j < n;++j)
        {
            std::cin >> k;
            if (k) pq.push(k);
            else if (!pq.empty()) {
                sum += pq.top(); pq.pop();
            }
        }
        std::cout << sum << "\n"; std::cout.flush(); 
    }
}   