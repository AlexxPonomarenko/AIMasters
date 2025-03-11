#include <iostream>
#include <map>
#include <set>

class TreapNode
{
public:
    TreapNode *left, *right, *parent;
    int64_t key, prior;
    size_t num_of_nodes;

    TreapNode(
        int64_t _k, int64_t _p = std::rand(), 
        TreapNode *_left = nullptr, TreapNode *_right = nullptr,
        TreapNode *_parent = nullptr
    ) {
        key = _k, prior = _p, left = _left, right = _right, parent = _parent, num_of_nodes = 0;
    }

    void update_data() {
        size_t s = 1;
        if (left) s += left->num_of_nodes;
        if (right) s += right->num_of_nodes;
        num_of_nodes = s;
    }

    void print() { std::cout << "(k: " << key << ", p: " << prior << ", n: " << num_of_nodes << ") "; }
    TreapNode* min() 
    {
        TreapNode* t = this;
        while (t) t->left;
        return t;
    }
    static TreapNode* find(TreapNode* root, int64_t key)
    {
        if (root->key == key)
            return root;
        else if (key > root->key && root->right)
            return find(root->right, key);
        else if (key < root->key && root->left)
            return find(root->left, key);
        else
            return nullptr;
    }
};

class Treap
{
    void __del(TreapNode *t) {
        if (!t) {
            return;
        }
        __del(t->left);
        __del(t->right);
        delete t;
    }

public:
    TreapNode *root;
    size_t sz;
    Treap() { root = nullptr; sz = 0; }
    ~Treap() { __del(root); }

    bool is_empty() { return !sz; }
    void _print(TreapNode* cur) { 
        if (!cur) return;
        if (cur->left) _print(cur->left);
        cur->print();
        if (cur->right) _print(cur->right);
    }
    void print() { _print(root); std::cout << "\n"; }

    void insert(int64_t k)
    {
        TreapNode *n = new TreapNode(k);
        sz += 1;

        if (!root) {
            root = n;
            return;
        }
        
        auto [t1, t2] = split(root, k);
        root = merge(t1, merge(n, t2));
    }

    std::pair<TreapNode*, TreapNode*> split(TreapNode *t, int64_t k)
    {
        if (!t) {
            return {nullptr, nullptr};
        }
        if(k == 0){
            return {nullptr,t}; 
        }
        auto tmp = 0;
        if (t->left) tmp = t->left->key;

        if (tmp < k) {
            auto [t1, t2] = split(t->right, k - 1 - tmp);
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

    
    TreapNode* find(int64_t key) { return TreapNode::find(root, key); }

    void remove(int64_t key)
    {
        auto [t1, t2] = split(root, key);
        TreapNode* removed_node = TreapNode::find(t2, key);
        TreapNode* father_removed_node = removed_node->parent;
        
        if (father_removed_node) {
            father_removed_node->left = removed_node->right;
            root = merge(t1, t2);
        }
        else {
            root = merge(t1, t2->right);
        }

        // root->parent = nullptr;
        sz--;
        delete removed_node;
    }
    TreapNode* remove_new(TreapNode* t, int key){
        if (!t){
            return nullptr;
        }
        if(key > t->key){
            t->right = remove_new(t->right,key);
            return t;
        } else if(key == t->key){
            TreapNode* r = t->right;
            TreapNode* l = t->left;
            delete t;
            t = merge(l,r);
            return t;
        } else{
            t->left = remove_new(t->left,key);
            return t;
        }
    }

    int64_t _k_order_stats(TreapNode* cur, size_t k)
    {
        size_t l_len = 0, r_len = 0;
        if (cur->left) l_len = cur->left->num_of_nodes;
        if (cur->right) r_len = cur->right->num_of_nodes;

        if (l_len + 1 == k) return cur->key;
        else if (cur->left && l_len >= k) return _k_order_stats(cur->left, k); 
        else if (cur->right && r_len < k) return _k_order_stats(cur->right, k - 1 - l_len);
    }
    int64_t k_order_stats(size_t k) { return _k_order_stats(root, k); }

    int64_t top() { return root->key; }
};  


void solve_treap(uint32_t n, uint32_t p)
{
    Treap t;
    for (int i = 1;i <= n;++i)
        t.insert(i);

    for (int i = 0;i < n;i++)
    {
        auto to_del = p % (n - i);
        if (to_del == 0) to_del = n - i;

        auto [t1, t2] = t.split(t.root, to_del);
        auto [f1, f2] = t.split(t1, to_del - 1);
        if (f2)
            std::cout << f2->key << " "; std::cout.flush();
            delete f2;
            t.root = t.merge(t2, f1);
        else
            std::cout << f2->key << " "; std::cout.flush();
            delete f2;
    }
}

int main()
{
    std::ios::sync_with_stdio(0); std::cin.tie(0);
    std::uint32_t n, p;
    std::cin >> n >> p;
    solve_treap(n, p);
    
}