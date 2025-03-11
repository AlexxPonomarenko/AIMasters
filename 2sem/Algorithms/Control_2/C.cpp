#include<iostream>
#include<vector>
#include<deque>
#include<algorithm>
#include<cassert>

using namespace std;

#define MAX_COST 1000000001

struct vertex{
	vertex(){}
};

class Graph{
	int n; // Number of vertices ≤ 10^5
public: 
	vector<vertex> V;
	Graph(int nn);
	//void printV();
	void printE();
	void printW();
	vector<vector<pair<int, uint32_t> > > neighbours;	
};



Graph::Graph(int nn){
	n = nn;
	vector<vertex> VV(n, vertex());
	V=VV;	
	neighbours = vector<vector<pair<int, uint32_t> > >(n, vector<pair<int, uint32_t> >());
}

	
struct Dijkstra{
	Graph *G;
	int n;
	int s;
	uint32_t M1;
	
	vector<uint32_t> dist;
	vector<int> pred;
	deque<pair<uint32_t, int> > q;
	
	Dijkstra(Graph *GG, int ss){
		M1=-1;
		G = GG;
		n = G->V.size();
		s = ss;
		dist = vector<uint32_t>(n, M1);
		pred = vector<int>(n, -1);
		dist[s] = 0;
		q.emplace_back(0, s); // heap.push_back(make_pair(0, s));
	}	
	
	void operator ()();
	
	void Relax(int u, int v, int w){
		if( dist[v] > dist[u] + w ){ 
			dist[v] = dist[u] + w; 
			pred[v] = u;
			q.emplace_back(dist[v], v);			
		}
	}
	
	void print_dist(){
		cout << "dist: " << endl;
		for(size_t i = 0; i < n; ++i){
			
			cout << "i: " << i << ", d[i]: ";
			 (dist[i] == M1) ? cout << -1 : cout << dist[i];
			 cout << endl;
		}
	}
	
};

void Dijkstra::operator ()(){				
    //cout << "here";
    while(  !q.empty() ){ // O(|E|)			
        auto [d, u] = q.front(); q.pop_front(); // O(|E|)
        
        assert(dist[u] <= d);
        if(dist[u] == d){ // только при закрытии вершины, т.е. O(|V|) запусков
            for(auto [v, w] : G->neighbours[u])
                Relax(u, v, w); 										
        }						
    }			
}

int main (int argc, char const *argv[])
{
	std::ios::sync_with_stdio(false); std::cin.tie(0);	
		
	uint32_t n, m;
	uint32_t M1 = -1;
	
	cin >> n >> m; 
    std::vector<uint32_t> color(n, 0);
    
    for (int i = 0;i < n;++i) {
        std::cin >> color[i];
    }
	
	Graph G(n);
	
	for(size_t i = 0; i < m; ++i){
		uint32_t u, v;
        uint32_t wf = 0, wt = 0;

		cin >> u >> v;
		--u; --v;

        if (color[u] != color[v]) {
            wf = 1 , wt = 1;
        }
        if (u % 2 == 1) {
            wf *= 2;
        }
        if (v % 2 == 1) {
            wt *= 2;
        }
		G.neighbours[u].emplace_back(v, wf);
		G.neighbours[v].emplace_back(u, wt);
	}

	
	Dijkstra dijkstra(&G, 0);
	
	dijkstra();
	
	if (dijkstra.dist[n - 1] != M1){ 
        std::vector<uint32_t> s_t_path;
        for (int i = n - 1; i != -1; i = dijkstra.pred[i]) {
            s_t_path.push_back(i);
        }
        std::cout << dijkstra.dist[n - 1] << " " <<  s_t_path.size() << "\n";
        while (s_t_path.size()) {
            std::cout << s_t_path.back() + 1 << " ";
            s_t_path.pop_back();
        }
        std::cout << std::endl;
	}
	else { std::cout << "impossible\n"; }	
		
	return 0;
}