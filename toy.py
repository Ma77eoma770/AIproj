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
        self.G.add_node(newid, name=newname, heuristic = max(1, old_heuristic - 2), is_bait=True)
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

            self.G.add_node(newid, name=newname, heuristic=max(1, old_heuristic - 2), is_bait=True)
            
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

            self.G.add_node(newid, name=newname, heuristic=max(1, old_heuristic - 2), is_bait=True)
            
            if i == 0:
                self.G.add_edge(start_node,newid)
            else:
                self.G.add_edge(newid-1,newid)

    def save_graph_png(self, filename="grafo_progetto.png"):
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        
        # 1. Definiamo i colori dei nodi in base alle loro proprietà e le etichette
        node_colors = []
        labels = {}
        
        for node, data in self.G.nodes(data=True):
            # Recuperiamo l'euristica (arrotondata a 1 decimale per pulizia visiva)
            heuristic_val = data.get('heuristic', 0.0)
            
            # Salviamo il nome e l'euristica su due righe da stampare sul nodo
            labels[node] = f"{data['name']}\nH: {heuristic_val:.1f}"
            
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
        
        # Disegniamo le etichette (font leggermente più piccolo per far spazio all'euristica)
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=7, font_family="sans-serif", font_weight="bold")
        
        # Personalizzazioni estetiche della figura
        plt.title("Rappresentazione del Grafo di Gioco (Tarpit)", fontsize=14, fontweight='bold')
        plt.axis("off") # Nasconde gli assi cartesiani standard
        
        # Salva l'immagine
        plt.savefig(filename, format="PNG", dpi=300, bbox_inches='tight')
        plt.close() # Chiude la figura per liberare memoria
        print(f"Grafo salvato con successo in '{filename}'")
    
    def get_neighbors(self, node):
        return list((n, self.G.nodes[n]['heuristic']) for n in self.G.successors(node))

    
    def get_neighbors_names(self, node):
        return [self.G.nodes[n]['name'] for n in self.G.successors(node)]

class environment():
    def __init__(self,pathlength):
        self.G = graph(pathlength)
        self.reset()
    
    def reset(self):
        self.G.reset()
        self.sim_on = True
        self.att = Attaccante(self.G.start_node, self.G.G.nodes[self.G.start_node]['heuristic'], max_patience=100)
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
        # Costi vivi delle azioni per evitare spam incontrollato
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
        
        if hasattr(self.att, 'previous_heuristic') and hasattr(self.att, 'current_heuristic'):
            h_prev = self.att.previous_heuristic
            h_curr = self.att.current_heuristic
            if (h_prev - h_curr) > 0:
                self.extimatedPatience = min(100.0, self.extimatedPatience + 5)
            else:
                self.extimatedPatience = max(0.0, self.extimatedPatience - 15)
                # IL VERO PREMIO: 30 punti ogni volta che l'attaccante va a sbattere!
                reward += 30.0  

        # Aggiornamento dello stato osservabile
        if att_status[0] == 'Normal Step':
            try:
                self.distanceFromStart = nx.shortest_path_length(self.G.G, source=self.G.start_node, target=self.att.current_node) 
            except nx.NetworkXNoPath:
                self.distanceFromStart = 0
            try:
                self.distanceToEnd = nx.shortest_path_length(self.G.G, source=self.att.current_node, target=self.G.target_node)
            except nx.NetworkXNoPath:
                self.distanceToEnd = 1000
            # RIMOSSO "reward += 5". L'IA ora deve SUDARE per fare punti usando Deadend e Loop!
        
        if att_status[0] == 'Backtracked':
            try:
                self.distanceFromStart = nx.shortest_path_length(self.G.G, source=self.G.start_node, target=self.att.current_node) 
            except nx.NetworkXNoPath:
                self.distanceFromStart = 0
            try:
                self.distanceToEnd = nx.shortest_path_length(self.G.G, source=self.att.current_node, target=self.G.target_node)
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
        
        return (
            self.distanceFromStart, 
            self.distanceToEnd, 
            int(self.extimatedPatience)
        )
    
    def get_neighbors(self,current_node):
        return self.G.get_neighbors(current_node)
    
    def checkTarget(self, node):
        return node == self.G.target_node
        
class Attaccante():
    def __init__(self, start_node, start_heuristic, max_patience=100):
        self.frontier = []      # Frontiera
        self.explored_nodes = set([start_node])  # Esplorati
        self.g_scores = {start_node: 0.0}
        self.patience = max_patience
        self.max_patience = max_patience
        self.current_node = start_node

        self.step_count = 0
        
        self.local_graph = nx.DiGraph()
        
        self.local_graph.add_node(start_node, heuristic=start_heuristic)
        self.current_heuristic = start_heuristic
        self.previous_heuristic = start_heuristic

    def step(self, env):
        self.step_count += 1
        
        # 1. ESPANSIONE (L'attaccante si guarda intorno ORA e vede le trappole appena piazzate)
        g_curr = self.g_scores[self.current_node]
        visible_neighbors = env.get_neighbors(self.current_node)
        
        for neighbor, h_neighbor in visible_neighbors:
            if neighbor not in self.local_graph:
                self.local_graph.add_node(neighbor, heuristic=h_neighbor)
                self.local_graph.add_edge(self.current_node, neighbor)
            
            # Ignoriamo i rami vecchi
            if neighbor in self.explored_nodes:
                continue
                
            if g_curr + 1 < self.g_scores.get(neighbor, float('inf')):
                self.g_scores[neighbor] = g_curr + 1
                f_neighbor = g_curr + 1 + h_neighbor
                hq.heappush(self.frontier, (f_neighbor, g_curr + 1, neighbor, self.patience, self.step_count, self.current_node))
        
        # Se siamo chiusi in un vicolo cieco senza alternative
        if not self.frontier:
            return ('Backtracked', self.current_node)

        # 2. SCELTA DELLA PROSSIMA MOSSA
        f_score, g_next, next_node, snapshot_patience, step_added, previous_node = hq.heappop(self.frontier)
        
        self.patience = snapshot_patience
        h_curr = self.local_graph.nodes[next_node]['heuristic']
        self.current_heuristic = h_curr
        
        if previous_node is not None and previous_node in self.local_graph.nodes:
            self.previous_heuristic = self.local_graph.nodes[previous_node]['heuristic']
        else:
            self.previous_heuristic = h_curr

        # 3. CONTROLLO PAZIENZA (si frustra se non fa progressi spaziali)
        if previous_node is not None and previous_node in self.local_graph.nodes:
            h_prev = self.local_graph.nodes[previous_node]['heuristic']
            if (h_prev - h_curr) > 0:
                self.patience = min(self.max_patience, self.patience + 5)
            else:
                self.patience = max(0, self.patience - 15)             
        
        if self.patience <= 0:
            self.backtrack() 
            return ('Backtracked', self.current_node)

        # 4. SPOSTAMENTO EFFETTIVO
        self.explored_nodes.add(next_node)
        self.current_node = next_node
        
        if env.checkTarget(self.current_node):
            return ('Target Reached', self.current_node)
            
        return ('Normal Step', self.current_node)

    def backtrack(self):
        new_frontier = []
        for f, g, node, patience, step_count, previous_node in self.frontier:
            if node not in self.explored_nodes and step_count < self.step_count:
                new_frontier.append((f, g, node, patience, step_count, previous_node))
        self.frontier = new_frontier
        hq.heapify(self.frontier)
        
        best_f,best_g,best_node,best_patience,best_step_count,best_previous_node = self.frontier[0]
        self.current_node = best_previous_node
        self.patience = best_patience
        self.step_count = best_step_count
    
    def checkPatience(self):
        return self.patience <= 0

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
    import random as rd
    
    # 1. Inizializziamo il difensore globale che manterrà la memoria tra i vari grafi
    difensore_in_training = Difensore(action_dimension=4, learning_rate=0.1, epsilon=0.9, gamma=0.95,epsilon_decay=0.9995, min_epsilon=0.01)
    
    NUM_EPISODI = 10000
    print(f"Inizio addestramento del Difensore Q-Learning (POMDP) per {NUM_EPISODI} episodi...\n")
    
    for episodio in range(1, NUM_EPISODI + 1):
        # Generiamo grafi di lunghezza variabile ad ogni episodio per favorire la generalizzazione
        lunghezza_percorso = rd.randint(5, 10)
        env = environment(pathlength=lunghezza_percorso)
        
        # Sostituiamo il difensore dell'ambiente con quello sotto addestramento
        env.dif = difensore_in_training
        
        env.reset()
        done = False
        total_reward = 0.0
        
        # Loop dell'episodio corrente
        while not done and env.sim_on:
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