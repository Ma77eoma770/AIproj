from random import random

import networkx as nx
import heapq as hq
import numpy as np

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
        
        # 1. Definiamo i colori dei nodi in base alle loro proprietà e le etichette
        node_colors = []
        node_sizes = []
        labels = {}
        
        for node, data in self.G.nodes(data=True):
            heuristic_val = data.get('heuristic', 0.0)
            base_label = f"{data['name']}\nH: {heuristic_val:.1f}"
            if node == attacker_node:
                labels[node] = f"{base_label}\nATTACKER"
            else:
                labels[node] = base_label
            
            if node == attacker_node:
                node_colors.append("#ff6b6b")
                node_sizes.append(2200)
            elif data['name'] == "Start":
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
        
        if attacker_node is not None:
            x, y = pos[attacker_node]
            ax.scatter([x], [y], s=500, color="white", edgecolor="#ff6b6b", linewidths=3, zorder=5)
            note = f"Attacker at {self.G.nodes[attacker_node]['name']}"
            if attacker_patience is not None:
                note += f" | patience {attacker_patience}"
            ax.text(
                x,
                y + 0.12,
                note,
                fontsize=8,
                fontweight="bold",
                color="#991b1b",
                ha="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ff6b6b", alpha=0.95),
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

class environment():
    def __init__(self,pathlength):
        self.G = graph(pathlength)
        self.reset()
    
    def reset(self):
        self.G.reset()
        self.sim_on = True
        self.att = Attaccante(self.G.start_node, self.G.G.nodes[self.G.start_node]['heuristic'])
        self.dif = Difensore()
        
        # Stato POMDP tracciato dall'ambiente
        self.distanceFromStart = 0
        self.distanceToEnd = self.G.pathlength - 1
        self.extimatedPatience = 100.0
        return self.getCurrentState()

    def step(self, training=True):
        """
        Esegue un passo combinato Difensore-Attaccante.
        Ritorna: (prev_state, action, reward, next_state, done)
        """

        curr_node = self.att.current_node
        # 1. Rileva lo stato parzialmente osservabile prima dell'azione
        prev_state = self.getCurrentState()
        
        # 2. Il difensore sceglie ed esegue l'azione basandosi sull'osservazione
        dif_action = self.dif.step(prev_state, training=training)
        
        reward = 0.0
        #Costi vivi delle azioni per evitare spam incontrollato
        if dif_action == 0:
            self.G.add_bait(curr_node)
            reward -= 1.0 
        elif dif_action == 1:
            self.G.add_loop(curr_node)
            reward -= 2.0
        elif dif_action == 2:
            self.G.add_deathend(curr_node)
            reward -= 2.0
        
        # 3. L'attaccante risponde muovendosi sul grafo modificato
        att_status = self.att.step(self)
        if att_status[0] == 'Stuck':
            self.sim_on = False
            print("L'attaccante è rimasto bloccato senza alternative. Simulazione terminata.")  

        undirected_G = self.G.G.to_undirected()

        # Aggiornamento dello stato osservabile
        if att_status[0] == 'Normal Step':
            try:
                self.distanceFromStart = nx.shortest_path_length(self.G.G, source=self.G.start_node, target=self.att.current_node) 
            except nx.NetworkXNoPath:
                self.distanceFromStart = 0
            try:
                self.distanceToEnd = nx.shortest_path_length(undirected_G, source=self.att.current_node, target=self.G.target_node)
            except nx.NetworkXNoPath:
                self.distanceToEnd = 1000
        
        if att_status[0] == 'Backtracked':
            try:
                self.distanceFromStart = nx.shortest_path_length(self.G.G, source=self.G.start_node, target=self.att.current_node) 
            except nx.NetworkXNoPath:
                self.distanceFromStart = 0
            try:
                self.distanceToEnd = nx.shortest_path_length(undirected_G, source=self.att.current_node, target=self.G.target_node)
            except nx.NetworkXNoPath:
                self.distanceToEnd = 1000
            
            # TRASFORMA LA PENALITÀ IN PREMIO MASSIMO: +50 per aver costretto l'attaccante a scappare!
            reward += 50.0  

        next_state = self.getCurrentState()
        done = False

        # 4. Condizioni di terminazione e ricompense finali
        if att_status[0] == 'Target Reached':
            shortest_path_length = nx.shortest_path_length(self.G.G, source=self.G.start_node, target=self.G.target_node)
            reward += 10 * (len(self.att.explored_nodes) - shortest_path_length + 1) # Reward proporzionale al numero di nodi esplorati oltre il percorso minimo (sprecati) 
            done = True
            self.sim_on = False

        return prev_state, dif_action, reward, next_state, done
    
    def getCurrentState(self):
        patience_bucket = int(self.extimatedPatience // 20)  # Bucket di pazienza da 0 a 10
        return (
            self.distanceFromStart, 
            self.distanceToEnd, 
            patience_bucket
        )
    
    def get_neighbors(self,current_node):
        return self.G.get_neighbors(current_node)
    
    def checkTarget(self, node):
        return node == self.G.target_node
        
class Attaccante():
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
        self.current_heuristic = self.local_graph.nodes[self.current_node]['heuristic']
        if self.previous_node is not None and self.previous_node in self.local_graph.nodes:
            h_prev = self.local_graph.nodes[self.previous_node]['heuristic']
            if (h_prev - self.current_heuristic) > 0:
                self.patience = min(self.max_patience, self.patience + 5)
            else:
                self.patience = max(0, self.patience - 15)             
        
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

class Difensore():
    def __init__(self, action_dimension=4, learning_rate=0.1, epsilon=0.1, gamma=0.9, epsilon_decay=0.9999, min_epsilon=0.01):
        self.action_dimension = action_dimension  # [0] esca, [1] loop, [2] vicolo cieco, [3] nessuna azione
        self.learning_rate = learning_rate  # Tasso di apprendimento
        self.epsilon = epsilon  # Fattore di esplorazione
        self.gamma = gamma  # Fattore di sconto
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table = {}  # Dizionario per memorizzare le Q-values

    def _get_q_values(self, state_tuple):
        """Inizializza pigramente i valori Q a 0.0 se lo stato è nuovo."""
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_dimension)
        return self.q_table[state_tuple]

    def step(self, state, training=True):
        state_tuple = tuple(state)

        if training and random() < self.epsilon:
            return np.random.randint(0, self.action_dimension)
        else:
            q_values = self._get_q_values(state_tuple)
        return np.argmax(q_values)
    
    def learn(self, state, action, reward, next_state, done):
        state_tuple = tuple(state)
        next_state_tuple = tuple(next_state)

        current_q = self._get_q_values(state_tuple)[action]
        
        if done:
            max_future_q = 0.0
        else:
            max_future_q = np.max(self._get_q_values(next_state_tuple))
            
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_future_q - current_q)
        
        self.q_table[state_tuple][action] = new_q

    def decay_epsilon(self):
        """Riduce l'esplorazione gradualmente, da chiamare a fine di ogni EPISODIO."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    

if __name__ == "__main__":
    # difensore = Difensore()
    # env = environment(15)
    # G = env.G
    # env.dif = difensore
    # step_count = 0
    # while env.sim_on:
    #     step_count += 1
    #     # env.G.save_graph_png(
    #     #     f"step_{step_count}.png",
    #     #     attacker_node=env.att.current_node,
    #     #     attacker_patience=env.att.patience,
    #     # )
    #     # print(f"Attaccante in {env.att.current_node} with patience {env.att.patience}")
    #     # step() esegue l'azione del difensore, la risposta dell'attaccante e calcola la ricompensa
    #     prev_state, action, reward, next_state, done = env.step(training=True)
        
    #     # Il difensore impara dall'esperienza parzialmente osservata
    #     difensore.learn(prev_state, action, reward, next_state, done)


    import random as rd
    
    # 1. Inizializziamo il difensore globale che manterrà la memoria tra i vari grafi
    difensore_in_training = Difensore(action_dimension=4, learning_rate=0.1, epsilon=0.9, gamma=0.95,epsilon_decay=0.9995, min_epsilon=0.01)
    
    NUM_EPISODI = 5000
    print(f"Inizio addestramento del Difensore Q-Learning (POMDP) per {NUM_EPISODI} episodi...\n")
    
    for episodio in range(1, NUM_EPISODI + 1):
        #print(f"\n=== EPISODIO {episodio} ===")
        # Generiamo grafi di lunghezza variabile ad ogni episodio per favorire la generalizzazione
        lunghezza_percorso = 9
        env = environment(pathlength=lunghezza_percorso)
        
        # Sostituiamo il difensore dell'ambiente con quello sotto addestramento
        env.dif = difensore_in_training
        
        env.reset()
        done = False
        total_reward = 0.0
        step_count = 0
        # Loop dell'episodio corrente
        while not done and env.sim_on:
            step_count += 1
            #env.G.save_graph_png(f"episodio_{episodio}_step_{step_count}.png")
            # step() esegue l'azione del difensore, la risposta dell'attaccante e calcola la ricompensa
            prev_state, action, reward, next_state, done = env.step(training=True)
            
            # Il difensore impara dall'esperienza parzialmente osservata
            difensore_in_training.learn(prev_state, action, reward, next_state, done)
            
            total_reward += reward
            
        # Al termine dell'episodio riduciamo progressivamente il fattore di esplorazione random
        difensore_in_training.decay_epsilon()
        
        if episodio % 100 == 0 or episodio == 1:
            print(f"Episodio {episodio:4d}/{NUM_EPISODI} | Percorso Base: {lunghezza_percorso} nodi | Reward Totale: {total_reward:6.1f} | Epsilon: {difensore_in_training.epsilon:.3f}")
            
    print("\nAddestramento Completato! La Q-table del difensore ha mappato", len(difensore_in_training.q_table), "stati.")
    print("\n" + "="*50)
    
    print("INIZIO SIMULAZIONE DI VALUTAZIONE (EPISODIO N+1)")
    print("="*50)
    
    # 1. Creiamo un ambiente dedicato per la simulazione finale
    lunghezza_test = 6  # Fissiamo una lunghezza per avere un output leggibile
    env_test = environment(pathlength=lunghezza_test)
    
    # Assegniamo il nostro difensore completamente addestrato
    env_test.dif = difensore_in_training
    env_test.reset()
    
    done = False
    step_sim = 0
    total_test_reward = 0.0
    
    # Mappatura delle azioni per una stampa leggibile
    action_names = {
        0: "Esca (Bait)", 
        1: "Loop", 
        2: "Vicolo Cieco (Deadend)", 
        3: "Nessuna Azione / Altro"
    }
    
    # Salviamo la foto iniziale del percorso base
    env_test.G.save_graph_png(f"simulazione_step_{step_sim}_inizio.png")
    
    while not done and env_test.sim_on:
        step_sim += 1
        print(f"\n--- TURNO {step_sim} ---")
        
        # Rileviamo la posizione attuale dell'attaccante
        att_node = env_test.att.current_node
        node_name = env_test.G.G.nodes[att_node]['name']
        print(f"📍 Posizione Attaccante: Nodo {att_node} [{node_name}]")
        
        # Rileviamo lo stato che viene passato alla rete
        curr_state = env_test.getCurrentState()
        print(f"📊 Stato POMDP (visto dal difensore): Distanza_Start={curr_state[0]}, Distanza_Target={curr_state[1]}, Pazienza_Stimata={curr_state[2]}")
        
        # IL DIFENSORE SCEGLIE L'AZIONE (training=False impone l'Exploit puro, senza mosse random)
        chosen_action = env_test.dif.step(curr_state, training=False)
        print(f"🛡️  Azione scelta dalla Q-Table: {action_names.get(chosen_action, 'Sconosciuta')} (Azione {chosen_action})")
        
        # Facciamo avanzare l'ambiente di uno step
        # Passiamo training=False anche qui per coerenza
        prev_state, action, reward, next_state, done = env_test.step(training=False)
        total_test_reward += reward
        
        # Mostriamo il risultato dell'azione sull'attaccante
        print(f"💥 Reward di questo turno: {reward}")
        print(f"😡 Pazienza REALE dell'attaccante: {env_test.att.patience}")
        print(f"🗺️  Frontiera attuale dell'A*: {[n[2] for n in env_test.att.frontier]}")
        
        # Scattiamo la foto del grafo dopo le modifiche
        nome_foto = f"simulazione_step_{step_sim}.png"
        env_test.G.save_graph_png(nome_foto)
        # La funzione save_graph_png ha già un suo print integrato[cite: 3]
        
    print("\n" + "="*50)
    print("SIMULAZIONE TERMINATA")
    print("="*50)
    print(f"🏆 Reward Totale accumulato: {total_test_reward}")
    print(f"🔍 Nodi totali esplorati (e sprecati) dall'attaccante: {len(env_test.att.explored_nodes)}")