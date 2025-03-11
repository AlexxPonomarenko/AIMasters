#include <iostream>
#include <vector>

# define MAXN 10000

void print_array(const std::vector<uint32_t> &v, size_t n)
{
    for (int j = 0;j < n - 1;++j) {
        std::cout << v[j] << " ";
        std::cout.flush();
    } 
    std::cout << v[n - 1] << "\n";
    std::cout.flush();
}

size_t __pivot(std::vector<uint32_t> &v, size_t l, size_t r)
{   
    uint32_t p = v[r - 1];
    int64_t i = l - 1;
    
    for(int64_t j = l;j < r;j++)
    {
        if (v[j] < p) std::swap(v[++i], v[j]);
    }
    std::swap(v[++i], v[r - 1]);
    return i;
}

void __qsort(std::vector<uint32_t> &v, size_t l, size_t r)
{
    if (r - l <= 1) {
        return;
    }
    size_t k = __pivot(v, l, r);
    __qsort(v, l, k);
    __qsort(v, k + 1, r);
}

void quicksort(std::vector<uint32_t> &v, size_t n)
{
    __qsort(v, 0, n);
}

int main()
{
    int n, t;
    std::vector<uint32_t> v;
    v.reserve(MAXN);

    std::cin >> t;

    for (int i = 0;i < t;++i)
    {
        std::cin >> n;
        for (int j = 0;j < n;++j)
        {
            uint32_t num;
            std::cin >> num;
            v[j] = num;
        }
        
        quicksort(v, n);
        print_array(v, n);
    }

}