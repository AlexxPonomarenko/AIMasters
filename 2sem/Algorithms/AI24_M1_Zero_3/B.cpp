#include <iostream>

#define N_MAX 1000000000

class TreapNode
{
public:
    TreapNode *left, *right;
    int64_t key, prior;

    TreapNode(int64_t _k, int64_t _p = std::rand() % 300000, TreapNode *_left = nullptr, TreapNode *_right = nullptr) {
        key = _k, prior = _p, left = _left, right = _right;
    }
};

class Treap
{
    TreapNode *root;
    void __del(TreapNode *t) {
        if (!t) {
            return;
        }
        __del(t->left);
        __del(t->right);
        delete t;
    }

public:
    Treap() { root = nullptr; }
    ~Treap() { __del(root); }

    void insert(int64_t k)
    {
        TreapNode *n = new TreapNode(k);
        if (!root) {
            root = n;
            return;
        }
        
        auto [t1, t2] = split(root, k);
        TreapNode *t = t2;
        if (t) {
            while (t->left) t = t->left;
            if (t->key == k) {
                root = merge(t1, t2);
                return;
            }
        }

        t1 = merge(t1, n);
        root = merge(t1, t2);
    }

    std::pair<TreapNode*, TreapNode*> split(TreapNode *t, int64_t k)
    {
        if (!t) {
            return {nullptr, nullptr};
        }
        if (t->key < k) {
            auto [t1, t2] = split(t->right, k);
            t->right = t1;
            return {t, t2};
        }
        else {
            auto [t1, t2] = split(t->left, k);
            t->left = t2;
            return {t1, t};
        }
    }

    TreapNode* merge(TreapNode *t1, TreapNode *t2)
    {
        if (!t2) return t1;
        if (!t1) return t2;

        if (t1->prior > t2->prior) {
            t1->right = merge(t1->right, t2);
            return t1;
        }
        else {
            t2->left = merge(t1, t2->left);
            return t2;
        }
    }

    int64_t upper_bound(int64_t i)
    {
        auto [t1, t2] = split(root, i);
        TreapNode *t = t2;
        if (t) {
            while (t->left) t = t->left;
            root = merge(t1, t2);
            return t->key;
        } 
        return -1;
    }
};  


int main()
{
    std::srand(666);

    size_t n;
    std::cin >> n;

    Treap t;
    char pred_op = 0;
    int64_t pred_ans;

    for (int i = 0;i < n;++i)
    {
        char op;
        int64_t z;
        std::cin >> op >> z;

        if (op == '+')
        {
            if (pred_op == '+' || !pred_op) {
                t.insert(z);
            } 
            else {
                t.insert((z + pred_ans) % N_MAX);
            }
        } else {
            pred_ans = t.upper_bound(z);
            std::cout << pred_ans << "\n"; std::cout.flush();
        }
        pred_op = op;
    }

}