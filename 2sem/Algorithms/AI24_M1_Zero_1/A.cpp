#include <iostream>

#define N_MAX 10000
#define T_MAX 1_000_000

void 
print_array(uint32_t *A, uint32_t n)
{
    for (uint32_t i = 0;i < n;++i)
    {
        std::cout << A[i] << " ";
        // std::cout.flush();
    }
    std::cout << "\n";
    std::cout.flush();
}

void
__merge(uint32_t *A, std::pair<uint32_t, uint32_t> A_bounds, std::pair<uint32_t, uint32_t> B_bounds)
{
    uint32_t A_point = A_bounds.first, B_point = B_bounds.first;
    uint32_t n = (B_bounds.second - B_bounds.first) + (A_bounds.second - A_bounds.first);
    
    uint32_t *out_arr = (uint32_t*)std::malloc(sizeof(uint32_t) * n);
    uint32_t out_point = 0;

    while (A_point != A_bounds.second || B_point != B_bounds.second)
    {
        uint32_t a = A[A_point], b = A[B_point];
        if (a < b)
        {
            if (A_point < A_bounds.second) {
                out_arr[out_point++] = A[A_point++];
            }
            else {
                out_arr[out_point++] = A[B_point++];
            }
        }
        else
        {
           if (B_point < B_bounds.second) {
                out_arr[out_point++] = A[B_point++];
            }
            else {
                out_arr[out_point++] = A[A_point++];
            } 
        }
    }

    for (uint32_t i = 0;i < n;++i)
    {
        A[i + A_bounds.first] = out_arr[i];
    }
    std::free(out_arr);
}

void
__merge_sort(uint32_t *A, uint32_t left, uint32_t right)
{
    uint32_t n = right - left;
    if (n < 2) {
        return;
    }
    __merge_sort(A, left, left + n / 2);
    __merge_sort(A, left + n / 2, right);
    __merge(A, std::pair<uint32_t, uint32_t>(left, left + n / 2), std::pair<uint32_t, uint32_t>(left + n / 2, right));
    
}

void 
merge_sort(uint32_t *A, uint32_t n)
{
    __merge_sort(A, 0, n);
}


void 
test()
{
    uint32_t A[9] = {1, 3, 6, 8, 9, 2, 5, 8, 9};
    __merge(A, std::pair<uint32_t, uint32_t>(0, 5), std::pair<uint32_t, uint32_t>(5, 9));
    print_array(A, 9); 
}

int
main()
{
    // std::freopen("t1.txt", "r", stdin);

    uint32_t t, n = 0;
    uint32_t A[N_MAX+1] = {};
    
    std::cin >> t;

    for (uint32_t c1 = 0;c1 < t; ++c1)
    {
        std::cin >> n;
        for (uint32_t i = 0;i < n;++i)
        {
            std::cin >> A[i];
        }
        merge_sort(A, n);
        print_array(A, n);
    }

    return 0;
}