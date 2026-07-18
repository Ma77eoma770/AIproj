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

        self.G.add_node(newid, name=newname, heuristic = 1, is_bait=True)
        self.G.add_edge(start_node, newid)
        return newid
    
    def add_loop(self, start_node, loop_length=3):
        for i in range(loop_length):
            newid = self.node_counter
            self.node_counter += 1

            newname= self.G.nodes[start_node]['name'] + f"_loop_{i}"
            
            self.G.add_node(newid, name=newname, heuristic=1, is_bait=True)
            
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
            
            self.G.add_node(newid, name=newname, heuristic=1, is_bait=True)
            
            if i == 0:
                self.G.add_edge(start_node,newid)
            else:
                self.G.add_edge(newid-1,newid)

    def save_graph_png(self, filename="grafo_progetto.png"):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        
        # 1. Definiamo i colori dei nodi in base alle loro proprietà
        node_colors = []
        labels = {}
        
        for node, data in self.G.nodes(data=True):
            # Salviamo il nome da stampare sul nodo
            labels[node] = data['name']
            
            # Coloriamo in modo diverso a seconda del tipo di nodo
            if data['name'] == "Start":
                node_colors.append("lightgreen")
            elif data['name'] == "Target":
                node_colors.append("gold")
            elif data.get('is_bait', False):
                node_colors.append("salmon") # Esche in rosso/arancio
            else:
                node_colors.append("skyblue") # Nodi normali del percorso
        
        # 2. Definiamo il layout (posizionamento dei nodi)
        # spring_layout simula forze fisiche per distanziare bene i rami e i cicli
        pos = nx.spring_layout(self.G, seed=42) 
        
        # 3. Disegniamo il grafo
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=1200, edgecolors="black")
        nx.draw_networkx_edges(self.G, pos, arrowstyle="->", arrowsize=15, edge_color="gray", width=1.5)
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8, font_family="sans-serif", font_weight="bold")
        
        # Personalizzazioni estetiche della figura
        plt.title("Rappresentazione del Grafo di Gioco (Tarpit)", fontsize=14, fontweight='bold')
        plt.axis("off") # Nasconde gli assi cartesiani standard
        
        # Salva l'immagine
        plt.savefig(filename, format="PNG", dpi=300, bbox_inches='tight')
        plt.close() # Chiude la figura per liberare memoria
        print(f"Grafo salvato con successo in '{filename}'")


class environment():
    def __init__(self,pathlength):
        self.G = graph(pathlength)
        self.reset()
    
    def reset(self):
        self.G.reset()
        self.sim_on = True
        self.att = Attaccante()
        self.dif = Difensore()
        self.extimatedPatience = 100

    def getCurrentState(self):
        return (self.G.current_node, self.G.distanceFromStart, self.G.distanceToEnd, self.extimatedPatience)
    
    def step(self):
        current_state = self.getCurrentState()
        
        dif_action = self.att.step(current_state)   #int azione che vuole si faccia dato lo stato attuale
        if dif_action == 0:
            self.G.add_bait(current_state[0])
        elif dif_action == 1:
            self.G.add_loop(current_state[0])
        elif dif_action == 2:
            self.G.add_deathend(current_state[0])

        att_action = self.dif.step(self.G, self.G.current_node)  #int nodo che vuole esplorare
        self.G.current_node = att_action

        ## Aggiornamento della extimatedPatience

        if self.att.checkPatience():   #Pazienza reale dell'attaccante ritorna true se si è stancato di esplorare
            self.sim_on = False
        


        


    

if __name__ == "__main__":
    env = graph(5)
    env.save_graph_png("init_state.png")
    env.add_deathend(2)
    env.save_graph_png("end_state.png")
    