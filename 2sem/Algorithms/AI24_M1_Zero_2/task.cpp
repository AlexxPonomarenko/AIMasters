#include <iostream>
#include <vector>
#include <string>

class Man
{
public:
    std::string name;
    uint64_t points;
    size_t t;

    Man(std::string n = "", uint64_t p = 0, size_t tt=0) {
        name = n; points = p; t = tt;
    }
    bool operator<(const Man &he) const
    {
        if (points == he.points) {
            return t > he.t;
        }
        return points < he.points;
    }
    bool operator==(const Man &he) const
    {
        return name == he.name;
    }
};

class PriorQueue
{
public:

    Man v[1000];
    size_t size;

    void print()
    {
        for (int i = 0;i < size - 1;++i)
        {
            std::cout << "{ " << v[i].name << " : " << v[i].points << " : " << v[i].t << " } --- ";
        }
        std::cout << "{ " << v[size - 1].name << " : " << v[size - 1].points << " : " << v[size - 1].t << " }\n";

    }

    void heapyfiy_up(size_t idx) {
        size_t parent_id = idx / 2;
        while (idx > 0) 
        {
            if (v[parent_id] < v[idx]) {
                std::swap(v[parent_id], v[idx]);
                idx = parent_id;
                parent_id = idx / 2;
            } else {
                break;
            }
        }
    }
    void heapyfiy_down(size_t idx) {
        size_t left = 2 * idx, right = 2 * idx + 1;
        size_t m = 0;
        while (left < size && right < size)
        {
            if (v[right] < v[left]) m = left;
            else m = right;

            if (v[idx] < v[m]) {
                std::swap(v[idx], v[m]);
                idx = m;
                left = 2 * idx, right = left + 1;
            }
            else break;
        }
        
    }
    int64_t find(const Man &m) {
        for (int i = 0;i < size;++i)
        {
            if (v[i].name == m.name) return i;
        }
        return -1;
    }

    PriorQueue(size_t m) { size = 0; }
    
    void change_prior(size_t idx, Man m)
    {
        if (m < v[idx]) return;
        v[idx] = m;
        heapyfiy_up(idx);
    }

    void insert(Man chel) {
        int64_t i = find(chel);
        if (i == -1) 
        {
            v[size++] = chel;
            heapyfiy_up(size - 1);
        }
        else change_prior(i, chel);
        // print();
    }
    
    Man extract_max() {
        Man ret = v[0];

        v[0] = v[--size];
        heapyfiy_down(0);
        return ret;
    }
};


int main()
{
    std::ios::sync_with_stdio(false); std::cin.tie(0);

    size_t m, n;
    std::cin >> m >> n;

    PriorQueue pq(m);

    int64_t p;
    std::string s;
    for (int i = 0;i < n;++i)
    {
        std::cin >> s >> p;
        Man m(s, p, i);
        pq.insert(m);
    }

    for (int i = 0;i < m;++i)
    {
        Man m = pq.extract_max();
        std::cout << m.name << "\n";
    }
}