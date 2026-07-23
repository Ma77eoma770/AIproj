import networkx as nx
import heapq as hq

class attaccante():
    def __init__(self, start_node, start_heuristic, max_patience=100, max_backtrack=3):
        self.start_node = start_node
        self.current_node = start_node
        self.previous_node = None
        self.current_heuristic = start_heuristic
        
        self.max_patience = max_patience
        self.patience = max_patience
        
        # Patience snapshots: node -> patience value when first explored
        self.patience_snapshots = {start_node: max_patience}
        
        # Backtracking / Retry limit
        self.backtrack_count = 0
        self.max_backtrack = max_backtrack
        
        # Search & Path representation
        self.frontier = []  # Heap of: (f_score, g_score, node, snapshot_patience, parent_node)
        self.explored_nodes = {start_node}
        self.failed_nodes = set()
        self.g_scores = {start_node: 0.0}
        
        self.local_graph = nx.DiGraph()
        self.local_graph.add_node(start_node, heuristic=start_heuristic)
        
        # Branch stagnation tracking
        self.best_heuristic_in_branch = start_heuristic
        self.stagnation_steps = 0

    def step(self, env):
        # 1. Target check
        if env.checkTarget(self.current_node):
            return ('Target Reached', self.current_node, None)
        
        # 2. Patience and Branch Inconsistency check
        if self.check_branch_inconsistency():
            self.patience = 0
            
        if self.patience <= 0:
            if self.backtrack_count >= self.max_backtrack:
                # self.log_stuck_situation(env, "Patience ran out and reached max backtrack count limit.")
                return ('GaveUp', self.current_node, None)
            
            success = self.backtrack(env)
            if not success:
                # self.log_stuck_situation(env, "Patience ran out, but backtracking failed to find any ancestor with unexplored neighbors.")
                return ('Stuck', self.current_node, None)
            
            return ('Backtracked', self.current_node, None)
        
        # 3. Expand neighbors of current node
        g_curr = self.g_scores[self.current_node]
        for neighbor, h_neighbor in env.get_neighbors(self.current_node):
            if neighbor in self.explored_nodes or neighbor in self.failed_nodes:
                continue
                
            if neighbor not in self.local_graph:
                self.local_graph.add_node(neighbor, heuristic=h_neighbor)
                self.local_graph.add_edge(self.current_node, neighbor)
                
            g_new = g_curr + 1
            if g_new < self.g_scores.get(neighbor, float('inf')):
                self.g_scores[neighbor] = g_new
                f_neighbor = g_new + h_neighbor
                # Push: (f_score, g_score, neighbor, self.patience, self.current_node)
                hq.heappush(self.frontier, (f_neighbor, g_new, neighbor, self.patience, self.current_node))
                
        # 4. Choose next step
        while self.frontier:
            f_score, g_next, next_node, snapshot_patience, parent_node = hq.heappop(self.frontier)
            if next_node not in self.explored_nodes and next_node not in self.failed_nodes:
                break
        else:
            # self.log_stuck_situation(env, "Frontier became empty (no unexplored and non-failed nodes left).")
            return ('Stuck', self.current_node)
            
        # 5. Move and calculate patience
        self.explored_nodes.add(next_node)
        h_next = self.local_graph.nodes[next_node]['heuristic']
        is_jump = (parent_node != self.current_node)
        
        if is_jump:
            # Jumping branches: restore the snapshot patience minus a penalty of 20
            self.patience = max(0, snapshot_patience - 20)
        else:
            # Normal step: lose 5 patience
            self.patience = max(0, self.patience - 5)
            
        # Save snapshot of patience if first visit to the node
        if next_node not in self.patience_snapshots:
            self.patience_snapshots[next_node] = self.patience
            
        # Update movement tracking state
        self.previous_node = self.current_node
        self.current_node = next_node
        self.current_heuristic = h_next
        
        status = 'Jumped' if is_jump else 'Normal Step'
        return (status, self.current_node, parent_node)

    def is_path_consistent(self, path):
        for i, node in enumerate(path):
            distance = len(path) - 1 - i
            h_val = self.local_graph.nodes[node]['heuristic']
            if distance > h_val + 3:
                return False
        return True

    def check_branch_inconsistency(self):
        """
        Traces the path from start to current node. If we went too far without reaching
        what the heuristic promised at some ancestor, we have branch inconsistency.
        """
        try:
            path = nx.shortest_path(self.local_graph, source=self.start_node, target=self.current_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False
        return not self.is_path_consistent(path)

    def backtrack(self, env):
        """
        Backtracks along the current branch to the first node with unexplored neighbors.
        """
        try:
            path = nx.shortest_path(self.local_graph, source=self.start_node, target=self.current_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return False
            
        # Trace path back from current_node to start_node to find backtrack target
        for i in range(len(path) - 1, -1, -1):
            node_on_path = path[i]
            # Ensure the prefix path up to this node is consistent (no inconsistency traps)
            if not self.is_path_consistent(path[:i+1]):
                continue
                
            # Check if this node has any unexplored neighbors (excluding failed nodes)
            neighbors = env.get_neighbors(node_on_path)
            has_unexplored = False
            for n, h in neighbors:
                if n not in self.explored_nodes and n not in self.failed_nodes:
                    has_unexplored = True
                    break
                    
            if has_unexplored and self.patience_snapshots.get(node_on_path, 0) > 0:
                # We found our backtrack target!
                # All descendants of the abandoned branch are marked failed
                if i + 1 < len(path):
                    abandoned_root = path[i+1]
                    abandoned_nodes = {abandoned_root} | nx.descendants(self.local_graph, abandoned_root)
                    for abandoned in abandoned_nodes:
                        self.failed_nodes.add(abandoned)
                        self.explored_nodes.add(abandoned) # ensure it is marked explored
                        self.g_scores[abandoned] = float('inf')
                    
                # Clean up the frontier of any failed nodes
                self.frontier = [item for item in self.frontier if item[2] not in self.failed_nodes]
                hq.heapify(self.frontier)
                
                # print(f"[ATTACKER] Backtracking from node {self.current_node} ({self.get_node_name(env, self.current_node)}) "
                #       f"to {node_on_path} ({self.get_node_name(env, node_on_path)}). Retry: {self.backtrack_count + 1}/{self.max_backtrack}")
                
                # Backtrack setting
                self.current_node = node_on_path
                self.patience = self.patience_snapshots.get(node_on_path, self.max_patience)
                self.backtrack_count += 1
                
                self.stagnation_steps = 0
                self.best_heuristic_in_branch = self.local_graph.nodes[node_on_path]['heuristic']
                self.current_heuristic = self.best_heuristic_in_branch
                
                if i > 0:
                    self.previous_node = path[i-1]
                else:
                    self.previous_node = None
                    
                return True
                
        # If no node on the path has unexplored neighbors, then the entire path is a dead-end
        for node in path:
            self.failed_nodes.add(node)
            self.g_scores[node] = float('inf')
        return False

    # def log_stuck_situation(self, env, reason):
    #     print(f"\n[ATTACKER DEBUG LOG] Stuck/GaveUp at node {self.current_node} ({self.get_node_name(env, self.current_node)})")
    #     print(f"Reason: {reason}")
    #     print(f"Patience: {self.patience}/{self.max_patience} | Backtrack Count: {self.backtrack_count}/{self.max_backtrack}")
        
    #     try:
    #         path = nx.shortest_path(self.local_graph, source=self.start_node, target=self.current_node)
    #         path_names = [f"{n} ({self.get_node_name(env, n)})" for n in path]
    #         print(f"Current Path: {' -> '.join(path_names)}")
    #     except Exception:
    #         print("Current Path: could not compute")
    #         path = []
            
    #     print("Ancestors neighbor status:")
    #     for n in reversed(path):
    #         neighbors = env.get_neighbors(n)
    #         unexplored = []
    #         explored = []
    #         failed = []
    #         for neighbor, h in neighbors:
    #             name = self.get_node_name(env, neighbor)
    #             if neighbor in self.failed_nodes:
    #                 failed.append(f"{neighbor} ({name})")
    #             elif neighbor in self.explored_nodes:
    #                 explored.append(f"{neighbor} ({name})")
    #             else:
    #                 unexplored.append(f"{neighbor} ({name})")
    #         print(f"  Node {n} ({self.get_node_name(env, n)}):")
    #         print(f"    - Unexplored: {unexplored}")
    #         print(f"    - Explored: {explored}")
    #         print(f"    - Failed: {failed}")

    #     frontier_nodes = [f"{item[2]} ({self.get_node_name(env, item[2])})" for item in self.frontier]
    #     print(f"Frontier (first 10): {frontier_nodes[:10]}")
    #     print(f"Explored nodes count: {len(self.explored_nodes)}")
    #     print(f"Failed nodes: {[f'{n} ({self.get_node_name(env, n)})' for n in self.failed_nodes]}")
    #     print("-" * 50 + "\n")

    def get_node_name(self, env, node):
        return env.G.get_node_name(node) if hasattr(env, 'G') else f"Node_{node}"
