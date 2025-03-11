    #include <iostream>
     
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
        size_t num_elements;
        Treap() { root = nullptr; num_elements = 0; }
        ~Treap() { __del(root); }
     
        bool is_empty() { return !num_elements; }
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
            num_elements += 1;
     
            if (!root) {
                root = n;
                return;
            }
            
            auto [t1, t2] = split(root, k);
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
     
                if (t1) t1->parent = t;
                t->update_data();
                
                return {t, t2};
            }
            else {
                auto [t1, t2] = split(t->left, k);
                t->left = t2;
     
                if (t2) t2->parent = t;
                t->update_data();
                
                return {t1, t};
            }
        }
     
        TreapNode* merge(TreapNode *t1, TreapNode *t2)
        {
            if (!t2) return t1;
            if (!t1) return t2;
     
            if (t1->prior > t2->prior) {
                t1->right = merge(t1->right, t2);
     
                t1->right->parent = t1;
                t1->update_data();
                
                return t1;
            }
            else {
                t2->left = merge(t1, t2->left);
     
                t2->left->parent = t2;
                t2->update_data();
                
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
            num_elements--;
            delete removed_node;
        }
     
        int64_t _k_order_stats(TreapNode* cur, size_t k)
        {
            size_t l_len = 0, r_len = 0;
            if (cur->left) l_len = cur->left->num_of_nodes;
            if (cur->right) r_len = cur->right->num_of_nodes;
     
            if (l_len + 1 == k) return cur->key;
            else if (cur->left && l_len >= k) return _k_order_stats(cur->left, k); 
            else if (cur->right && r_len < k) return _k_order_stats(cur->right, k - 1 - l_len);
     
            throw std::invalid_argument("Invalid k-order statistics");
        }
        int64_t k_order_stats(size_t k) { return _k_order_stats(root, k); }
     
        int64_t top() { return root->key; }
    };  
     
    int main()
    {
        // std::ios::sync_with_stdio(false); std::cin.tie(0);
        std::srand(0);
     
        uint64_t n, q;
        int64_t k;
     
        Treap qp;
     
        std::cin >> n >> q;
        for (int i = 0;i < n;++i)
        {
            std::cin >> k;
            qp.insert(k);
        }
     
        for (int i = 0;i < q;++i) 
        {
            std::cin >> k;
            int64_t k_stats;
            try {
                k_stats = qp.k_order_stats(std::abs(k));
            }
            catch (const std::exception& e) {
                continue;
            }
     
            if (1 <= k && k <= n) qp.insert(k);
            if (k < 0) qp.remove(k_stats); 
            // qp.print();
        }
     
        if (qp.is_empty()) std::cout << 0 << "\n";
        else std::cout << qp.top() << "\n";
    }