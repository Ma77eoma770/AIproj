import networkx as nx
import heapq as hq

class attaccante():
    def __init__(self, start_node, start_heuristic, max_patience=100):
        self.frontier = []      
        self.explored_nodes = set([start_node])  
        self.g_scores = {start_node: 0.0}
        self.max_patience = max_patience
        self.patience = max_patience
        self.current_node = start_node
        self.step_count = 0
        
        self.local_graph = nx.DiGraph()
        self.local_graph.add_node(start_node, heuristic=start_heuristic)
        
        self.previous_node = None
        self.current_heuristic = start_heuristic

        self.failed_nodes = set()  # Nodi che hanno portato a vicoli ciechi o loop

    def step(self, env):
        self.env = env
        
        if env.checkTarget(self.current_node):
            return ('Target Reached', self.current_node)
        
        self.step_count += 1

        # 1. CONTROLLO PAZIENZA             
        
        if self.patience <= 0:
            success = self.backtrack() 
            if not success:
                return ('Stuck', self.current_node)
            return ('Backtracked', self.current_node)
        
        # 2. ESPANSIONE
        g_curr = self.g_scores[self.current_node]
        visible_neighbors = env.get_neighbors(self.current_node)
        
        for neighbor, h_neighbor in visible_neighbors:
            if neighbor in self.explored_nodes or neighbor in self.failed_nodes:
                continue

            if neighbor not in self.local_graph:
                self.local_graph.add_node(neighbor, heuristic=h_neighbor)
                self.local_graph.add_edge(self.current_node, neighbor)
                
            if g_curr + 1 < self.g_scores.get(neighbor, float('inf')):
                self.g_scores[neighbor] = g_curr + 1
                f_neighbor = g_curr + 1 + h_neighbor
                hq.heappush(self.frontier, (f_neighbor, g_curr + 1, neighbor, self.patience, self.step_count, self.current_node))

        if not self.frontier:
            return ('Stuck', self.current_node)

        # 3. SCELTA DELLA PROSSIMA MOSSA (Con filtro anti-doppioni)
        # Poiché col backtracking l'agente ripassa per nodi vecchi, saltiamo le alternative già esplorate
        while self.frontier:
            f_score, g_next, next_node, snapshot_patience, step_added, prev_n = hq.heappop(self.frontier)
            if next_node not in self.explored_nodes:
                break
        else:
            # Se la frontiera si svuota scartando i doppioni
            return ('Stuck', self.current_node)

        # 4. SPOSTAMENTO EFFETTIVO
        self.explored_nodes.add(next_node)
        self.previous_node = self.current_node  
        self.current_node = next_node

        self.current_heuristic = self.local_graph.nodes[self.current_node]['heuristic']
        if self.previous_node is not None and self.previous_node in self.local_graph.nodes:
            h_prev = self.local_graph.nodes[self.previous_node]['heuristic']
            if (h_prev - self.current_heuristic) > 0:
                self.patience = min(self.max_patience, self.patience + 5)
            else:
                self.patience = max(0, self.patience - 15)
                
        return ('Normal Step', self.current_node)

    def backtrack(self):
        #print(f"Starting backtracking from node {self.get_node_name(self.current_node)} due to patience 0")
        
        nodo_fallito = self.current_node
        self.failed_nodes.add(nodo_fallito)
        self.explored_nodes.add(nodo_fallito)
        self.g_scores[nodo_fallito] = float('inf')
        
        # Potatura chirurgica sicura del ramo fallito
        nodi_da_potare = set(nx.descendants(self.local_graph, nodo_fallito))
        nodi_da_potare.add(nodo_fallito)
        
        for n in nodi_da_potare:
            self.explored_nodes.add(n)
            self.failed_nodes.add(n)
            self.g_scores[n] = float('inf')

        new_frontier = []
        for f, g, node, patience, step_count, previous_node in self.frontier:
            if node not in self.explored_nodes and node not in nodi_da_potare:
                new_frontier.append((f, g, node, patience, step_count, previous_node))
        
        self.frontier = new_frontier
        hq.heapify(self.frontier)
        
        if not self.frontier:
            print('No more alternatives left in the frontier. Attacker is stuck.')
            return False
            
        # CORREZIONE CHIAVE: Leggiamo l'alternativa ma NON la rimuoviamo (lo farà step al prossimo turno)
        best_f, best_g, best_node, best_patience, best_step_count, best_previous_node = self.frontier[0]
        
        # Riportiamo fisicamente l'agente sull'antenato (es. il nodo B) per riprendere il cammino
        self.current_node = best_previous_node
        self.patience = best_patience
        self.step_count = best_step_count
        
        # Ripristiniamo coerentemente il 'previous_node' guardando il grafo
        parents = list(self.local_graph.predecessors(best_previous_node))
        self.previous_node = parents[0] if parents else None
        
        #print(f"Backtracking complete. Returned to ancestor: {self.get_node_name(self.current_node)}")
        return True
    
    def checkPatience(self):
        return self.patience <= 0
    
    def get_node_name(self, node):
        return self.env.G.get_node_name(node) if hasattr(self, 'env') else f"Node_{node}"
