import networkx as nx

class graph:
    def __init__(self,pathlength):
        self.pathlength = pathlength
        self.reset()

    def reset(self):
        self.G = nx.DiGraph()
        self.node_counter = 0 
        for i in range(self.pathlength):
            node_id = self.node_counter
            # Definiamo il nome testuale in base al ruolo del nodo
            if i == 0:
                node_name = "Start"
            elif i == self.pathlength - 1:
                node_name = "Target"
            else:
                node_name = f"Node_{i}"
            self.G.add_node(
                node_id, 
                name=node_name,                               # Stringa descrittiva
                heuristic=float(self.pathlength - i - 1), 
                is_bait=False,
            )
            if i > 0:
                # Colleghiamo il nodo precedente (node_id - 1) a quello corrente
                self.G.add_edge(node_id - 1, node_id)
            self.node_counter += 1
        self.start_node = 0                     # ID del nodo Start
        self.target_node = self.pathlength - 1  # ID del nodo Target
    
    def add_bait(self, start_node):
        newid = self.node_counter
        self.node_counter += 1

        newname= self.G.nodes[start_node]['name'] + "_bait"

        old_heuristic = self.G.nodes[start_node]['heuristic']
        self.G.add_node(newid, name=newname, heuristic = max(1, old_heuristic - 1), is_bait=True)
        self.G.add_edge(start_node, newid)
        return newid
    
    def add_loop(self, start_node, loop_length=3):
        for i in range(loop_length):
            newid = self.node_counter
            self.node_counter += 1

            newname= self.G.nodes[start_node]['name'] + f"_loop_{i}"
            if i == 0:
                old_heuristic = self.G.nodes[start_node]['heuristic']
            else:
                old_heuristic = self.G.nodes[newid-1]['heuristic']

            self.G.add_node(newid, name=newname, heuristic=max(1, old_heuristic - 1), is_bait=True)
            
            if i == 0:
                self.G.add_edge(start_node,newid)
            else:
                self.G.add_edge(newid-1,newid)
        self.G.add_edge(newid,start_node)

    def add_deathend(self, start_node, deathend_length=3):
        for i in range(deathend_length):
            newid = self.node_counter
            self.node_counter += 1

            newname= self.G.nodes[start_node]['name'] + f"_deathend_{i}"
            
            if i == 0:
                old_heuristic = self.G.nodes[start_node]['heuristic']
            else:
                old_heuristic = self.G.nodes[newid-1]['heuristic']

            self.G.add_node(newid, name=newname, heuristic=max(1, old_heuristic - 1), is_bait=True)
            
            if i == 0:
                self.G.add_edge(start_node,newid)
            else:
                self.G.add_edge(newid-1,newid)

    def save_graph_png(self, filename="grafo_progetto.png", attacker_node=None, attacker_patience=None):
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 9))
        
        #1. Definiamo i colori dei nodi in base alle loro proprietà e le etichette
        node_colors = []
        node_sizes = []
        labels = {}
        
        for node, data in self.G.nodes(data=True):
            heuristic_val = data.get('heuristic', 0.0)
            # Etichetta normale, senza aggiungere "ATTACKER"
            labels[node] = f"{data['name']}\nH: {heuristic_val:.1f}"

            if data['name'] == "Start":
                node_colors.append("#8ee38b")
                node_sizes.append(1100)
            elif data['name'] == "Target":
                node_colors.append("#ffd166")
                node_sizes.append(1100)
            elif data.get('is_bait', False):
                node_colors.append("#f4a261")
                node_sizes.append(900)
            else:
                node_colors.append("#7ec8e3")
                node_sizes.append(800)
        
        # 2. Layout più pulito per grafi con molti nodi
        if self.G.number_of_nodes() > 0:
            start_node = getattr(self, "start_node", None)
            if start_node is None:
                start_node = next(iter(self.G.nodes()))

            layer_map = {start_node: 0}
            bfs_order = [start_node]
            queue = [start_node]

            while queue:
                current = queue.pop()
                for neighbor in self.G.successors(current):
                    if neighbor not in layer_map:
                        layer_map[neighbor] = layer_map[current] + 1
                        bfs_order.append(neighbor)
                        queue.append(neighbor)

            for node in self.G.nodes():
                if node not in layer_map:
                    layer_map[node] = max(layer_map.values(), default=-1) + 1

            layer_nodes = {}
            for node, level in layer_map.items():
                layer_nodes.setdefault(level, []).append(node)

            for nodes in layer_nodes.values():
                nodes.sort(key=lambda n: (self.G.nodes[n]['name'], n))

            max_layer = max(layer_nodes.keys(), default=0)
            pos = {}
            for level, nodes in sorted(layer_nodes.items()):
                x = (level - max_layer / 2.0) * 1.8
                if len(nodes) == 1:
                    ys = [0.0]
                else:
                    spacing = 1.4 / max(1, len(nodes) - 1)
                    ys = [-0.7 + i * spacing for i in range(len(nodes))]
                for index, node in enumerate(nodes):
                    pos[node] = (x, ys[index])
        else:
            pos = {}
        
        # 3. Disegniamo il grafo
        nx.draw_networkx_nodes(
            self.G,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="black",
            linewidths=1.2,
            ax=ax,
        )
        nx.draw_networkx_edges(
            self.G,
            pos,
            arrowstyle="->",
            arrowsize=15,
            edge_color="gray",
            width=1.5,
            alpha=0.85,
            ax=ax,
        )
        
        nx.draw_networkx_labels(
            self.G,
            pos,
            labels=labels,
            font_size=6,
            font_family="sans-serif",
            font_weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="0.7"),
            ax=ax,
        )
        
        if attacker_node is not None and attacker_node in pos:
            x, y = pos[attacker_node]
            
            # Prepariamo il testo dell'etichetta
            note = "Attacker"
            if attacker_patience is not None:
                note += f"\nPazienza: {attacker_patience}"
                
            # Aggiungiamo una freccia che punta al nodo
            # xy è la punta della freccia, xytext è dove si trova il riquadro col testo
            ax.annotate(
                note,
                xy=(x, y + 0.08),     # La punta della freccia sfiora la parte superiore del nodo
                xytext=(x, y + 0.45), # Posizione del riquadro di testo (più in alto)
                arrowprops=dict(facecolor='#d90429', edgecolor='#8d0801', shrink=0.02, width=3, headwidth=10),
                fontsize=9,
                fontweight="bold",
                color="white",
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.4", fc="#ef233c", ec="#8d0801", alpha=0.95),
                zorder=10 # Lo mettiamo in primo piano sopra a tutto il resto
            )
        
        ax.set_title("Rappresentazione del Grafo di Gioco (Tarpit)", fontsize=14, fontweight='bold')
        ax.axis("off")
        plt.tight_layout()
        
        # Salva l'immagine
        plt.savefig(filename, format="PNG", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafo salvato con successo in '{filename}'")
    
    def get_neighbors(self, node):
        return list((n, self.G.nodes[n]['heuristic']) for n in self.G.successors(node))

    
    def get_neighbors_names(self, node):
        return [self.G.nodes[n]['name'] for n in self.G.successors(node)]
    
    def get_node_name(self, node):
        return self.G.nodes[node]['name']