import networkx as nx
import random
import heapq

class GraphEnvironment:
    """
    Rappresenta l'ambiente di gioco basato su un grafo dinamico di NetworkX.
    Inizialmente composto da una linea di 5 nodi da 'Start' a 'Target'.
    """
    def __init__(self):
        self.G = nx.DiGraph()
        # Nodi iniziali sicuri della linea principale
        self.main_path = ['Start', 'N1', 'N2', 'N3', 'Target']
        
        # Inizializza i nodi con euristica decrescente (h) e profondità
        for i, node in enumerate(self.main_path):
            self.G.add_node(
                node, 
                secure=True, 
                depth=i, 
                h=4 - i  # h decresce verso il Target (Start: 4, Target: 0)
            )
            
        # Connette i nodi con archi orientati di peso iniziale 1
        for i in range(len(self.main_path) - 1):
            self.G.add_edge(self.main_path[i], self.main_path[i+1], weight=1)
            
        # Contatore per generare nodi esca/vicoli ciechi univoci
        self.node_counter = 0

    def add_dead_end(self, current_node):
        """
        Azione 1: Aggiunge un vicolo cieco composto da 2 nodi in sequenza (non sicuri)
        collegati al nodo corrente.
        """
        d1 = f"D{self.node_counter}"
        d2 = f"D{self.node_counter + 1}"
        self.node_counter += 2
        
        curr_depth = self.G.nodes[current_node]['depth']
        curr_h = self.G.nodes[current_node]['h']
        
        # I nodi hanno profondità crescente ma non portano al Target.
        # L'euristica decresce per trarre in inganno A*.
        self.G.add_node(d1, secure=False, depth=curr_depth + 1, h=max(0.5, curr_h - 0.5))
        self.G.add_node(d2, secure=False, depth=curr_depth + 2, h=max(0.5, curr_h - 1.0))
        
        self.G.add_edge(current_node, d1, weight=1)
        self.G.add_edge(d1, d2, weight=1)
        return d1, d2

    def add_bait(self, current_node):
        """
        Azione 2: Crea un nodo esca (unsecure) adiacente al nodo corrente 
        con un'euristica molto bassa per attrarre l'A*.
        """
        b = f"B{self.node_counter}"
        self.node_counter += 1
        
        curr_depth = self.G.nodes[current_node]['depth']
        
        # Euristica estremamente bassa (0.1) per renderlo prioritario rispetto alla via principale
        self.G.add_node(b, secure=False, depth=curr_depth + 1, h=0.1)
        self.G.add_edge(current_node, b, weight=1)
        return b

    def increase_edge_weight(self, current_node):
        """
        Azione 3: Incrementa il costo del cammino sulla via sicura principale,
        rendendo le deviazioni non sicure temporaneamente più attraenti.
        """
        neighbors = list(self.G.successors(current_node))
        secure_neighbors = [n for n in neighbors if self.G.nodes[n].get('secure', False)]
        if secure_neighbors:
            target_node = secure_neighbors[0]
            self.G[current_node][target_node]['weight'] = 5
            return target_node
        return None


class Attacker:
    """
    Rappresenta l'Attaccante che esegue una ricerca A* modificata un passo alla volta.
    Gestisce il Sospetto e il Backtracking in caso di superamento della pazienza.
    """
    def __init__(self, env, patience_threshold=30):
        self.env = env
        self.patience_threshold = patience_threshold
        self.reset_search()

    def reset_search(self):
        self.suspicion = 0
        self.max_depth_reached = 0
        self.open_list = []  # Heap di tuple: (f, counter, node, parent, g)
        self.closed_list = set()
        self.parent_map = {}
        self.blacklisted_nodes = set()
        self.current_node = 'Start'
        self.counter = 0
        
        # Aggiunge il nodo Start all'inizio
        h_start = self.env.G.nodes['Start']['h']
        heapq.heappush(self.open_list, (h_start, self.counter, 'Start', None, 0))
        self.counter += 1

    def step(self):
        """
        Esegue un singolo passo di espansione dell'A*.
        Ritorna una tupla: (nodo_selezionato, backtracked, status)
        """
        if not self.open_list:
            return None, False, 'stuck'
            
        # Trova il nodo con f minimo non ancora visitato e non blacklistato
        node = None
        parent = None
        g_val = 0
        while self.open_list:
            f, _, n, p, g = heapq.heappop(self.open_list)
            if n not in self.closed_list and n not in self.blacklisted_nodes:
                node = n
                parent = p
                g_val = g
                break
                
        if node is None:
            return None, False, 'stuck'
            
        # Segna il nodo come visitato
        self.closed_list.add(node)
        self.current_node = node
        self.parent_map[node] = parent
        
        # Condizione di vittoria dell'attaccante
        if node == 'Target':
            return node, False, 'won'
            
        # Controllo della profondità e del sospetto
        depth = self.env.G.nodes[node]['depth']
        backtracked = False
        
        if depth > self.max_depth_reached:
            self.max_depth_reached = depth
        else:
            self.suspicion += 10
            
        # Gestione del Backtracking
        if self.suspicion > self.patience_threshold:
            backtracked = True
            
            # Traccia a ritroso fino all'ultimo nodo sicuro
            curr = node
            while curr is not None and not self.env.G.nodes[curr].get('secure', False):
                curr = self.parent_map.get(curr)
            last_secure = curr if curr is not None else 'Start'
            
            # Blacklist dei nodi generati dal difensore per non ricadere nella trappola
            for n in list(self.env.G.nodes):
                if not self.env.G.nodes[n].get('secure', False):
                    self.blacklisted_nodes.add(n)
                    
            # Resetta lo stato di ricerca partendo dall'ultimo nodo sicuro
            self.open_list = []
            self.suspicion = 0
            
            # Ricalcola la g per il nodo sicuro di ripartenza
            parent_secure = self.parent_map.get(last_secure)
            g_secure = 0
            temp = last_secure
            while temp != 'Start' and temp is not None:
                p = self.parent_map.get(temp)
                if p is not None:
                    g_secure += self.env.G[p][temp].get('weight', 1)
                temp = p
                
            h_secure = self.env.G.nodes[last_secure]['h']
            heapq.heappush(self.open_list, (g_secure + h_secure, self.counter, last_secure, parent_secure, g_secure))
            self.counter += 1
            self.current_node = last_secure
        else:
            # Espansione standard dei vicini del nodo corrente
            for neighbor in self.env.G.successors(node):
                if neighbor in self.closed_list or neighbor in self.blacklisted_nodes:
                    continue
                weight = self.env.G[node][neighbor].get('weight', 1)
                g_new = g_val + weight
                h_new = self.env.G.nodes[neighbor]['h']
                f_new = g_new + h_new
                heapq.heappush(self.open_list, (f_new, self.counter, neighbor, node, g_new))
                self.counter += 1
                
        return node, backtracked, 'active'


class Defender:
    """
    Rappresenta il Difensore che apprende tramite Tabular Q-Learning.
    """
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}  # Q-Table: {(state, action): q_value}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_actions = 4  # 0: No-op, 1: Add_Dead_End, 2: Add_Bait, 3: Increase_Edge_Weight
        self.action_names = {
            0: "No-op",
            1: "Add_Dead_End",
            2: "Add_Bait",
            3: "Increase_Edge_Weight"
        }

    def get_state(self, depth, suspicion):
        # Discretizza il Sospetto in 3 livelli: 0 (Basso), 1 (Medio), 2 (Alto)
        if suspicion <= 10:
            susp_level = 0
        elif suspicion == 20:
            susp_level = 1
        else:
            susp_level = 2
        return (depth, susp_level)

    def choose_action(self, state):
        # Politica epsilon-greedy
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        
        # Scelta greedy
        q_values = [self.q_table.get((state, a), 0.0) for a in range(self.num_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, done):
        prev_q = self.q_table.get((state, action), 0.0)
        if done:
            target = reward
        else:
            max_next_q = max([self.q_table.get((next_state, a), 0.0) for a in range(self.num_actions)])
            target = reward + self.gamma * max_next_q
        self.q_table[(state, action)] = prev_q + self.alpha * (target - prev_q)


def run_episode(env, attacker, defender, is_training=False):
    """
    Esegue un singolo episodio di gioco e restituisce il log degli eventi.
    """
    logs = []
    turn = 1
    max_turns = 50
    
    attacker.reset_search()
    prev_state = None
    prev_action = None
    
    while turn <= max_turns:
        # 1. Fase Attaccante: determina il nodo di espansione corrente
        # Per poter permettere al difensore di agire sul nodo prima che venga espanso,
        # estraiamo il nodo in cima alla open_list.
        if not attacker.open_list:
            logs.append(f"Turno {turn}: L'Attaccante è bloccato (Open List vuota).")
            break
            
        # Troviamo il prossimo nodo valido senza rimuoverlo permanentemente ancora
        valid_node_found = False
        temp_open = sorted(attacker.open_list)
        next_node = None
        for item in temp_open:
            if item[2] not in attacker.closed_list and item[2] not in attacker.blacklisted_nodes:
                next_node = item[2]
                valid_node_found = True
                break
                
        if not valid_node_found:
            logs.append(f"Turno {turn}: L'Attaccante è bloccato (nessun nodo valido da espandere).")
            break
            
        # 2. Fase Difensore: Osserva lo stato ed agisce sul nodo che l'attaccante sta per espandere
        state = defender.get_state(env.G.nodes[next_node]['depth'], attacker.suspicion)
        action = defender.choose_action(state)
        
        # Esegui azione del difensore
        action_detail = ""
        if action == 0:
            action_detail = "Nessuna azione (No-op)"
        elif action == 1:
            d1, d2 = env.add_dead_end(next_node)
            action_detail = f"Aggiunto vicolo cieco {d1} -> {d2} da {next_node}"
        elif action == 2:
            b = env.add_bait(next_node)
            action_detail = f"Aggiunta esca {b} da {next_node}"
        elif action == 3:
            target_node = env.increase_edge_weight(next_node)
            if target_node:
                action_detail = f"Aumentato peso dell'arco {next_node} -> {target_node}"
            else:
                action_detail = "Tentato aumento peso (nessun vicino sicuro trovato, eseguito No-op)"
                action = 0 # Trattato come no-op
                
        # 3. Fase Attaccante: Espansione effettiva
        expanded, backtracked, status = attacker.step()
        
        # Calcolo reward per il difensore
        reward = -15 if backtracked else 1
        
        # Registra i log del turno
        logs.append(
            f"Turno {turn:02d} | "
            f"Attaccante in {expanded} (Sospetto: {attacker.suspicion}) | "
            f"Difensore Stato: {state} -> Azione: {defender.action_names[action]} ({action_detail}) | "
            f"Reward: {reward}"
        )
        
        if backtracked:
            logs.append(f"  >>> ATTENZIONE: L'Attaccante ha rilevato una trappola! Backtracking al nodo sicuro: {attacker.current_node}")
            
        # Aggiornamento Q-table per il turno precedente
        if prev_state is not None and is_training:
            defender.update_q_table(prev_state, prev_action, reward, state, False)
            
        prev_state = state
        prev_action = action
        
        # Controlla condizioni di fine episodio
        if status == 'won':
            logs.append(f"Fine Gioco: L'Attaccante ha raggiunto il Target '{expanded}' in {turn} turni. (Attaccante Vince!)")
            if is_training:
                defender.update_q_table(prev_state, prev_action, -20, state, True)
            break
        elif status == 'stuck':
            logs.append(f"Fine Gioco: L'Attaccante si è bloccato al turno {turn}. (Difensore Vince!)")
            if is_training:
                defender.update_q_table(prev_state, prev_action, 20, state, True)
            break
            
        turn += 1
        
    if turn > max_turns:
        logs.append(f"Fine Gioco: Raggiunto il limite di {max_turns} turni. (Difensore Vince!)")
        if is_training and prev_state is not None:
            defender.update_q_table(prev_state, prev_action, 20, state, True)
            
    return logs


if __name__ == "__main__":
    # Addestramento minimale del difensore per 500 episodi per fargli apprendere una policy base
    print("Addestramento del Difensore in corso per 500 episodi...")
    
    # Inizializziamo il difensore
    defender_agent = Defender(alpha=0.1, gamma=0.9, epsilon=0.2)
    
    for episode in range(500):
        env_instance = GraphEnvironment()
        attacker_agent = Attacker(env_instance)
        # Esegui episodio in modalità addestramento (aggiorna la Q-Table)
        run_episode(env_instance, attacker_agent, defender_agent, is_training=True)
        
    print("Addestramento completato.")
    print("-" * 80)
    
    # Eseguiamo un singolo episodio di prova con log dettagliato dei turni
    print("Esecuzione di un episodio di prova con la policy appresa:")
    env_test = GraphEnvironment()
    attacker_test = Attacker(env_test)
    
    # Disabilitiamo l'esplorazione randomica del difensore per il test
    defender_agent.epsilon = 0.0
    
    episode_logs = run_episode(env_test, attacker_test, defender_agent, is_training=False)
    for log in episode_logs:
        print(log)
        
    print("-" * 80)
    print("Q-Table del Difensore appresa (Stato: (profondità, sospetto) -> Azione):")
    for key, value in sorted(defender_agent.q_table.items()):
        state, action = key
        # Stampiamo solo le coppie stato-azione rilevanti con Q-value significativo
        if abs(value) > 0.01:
            print(f"Stato {state} | Azione: {defender_agent.action_names[action]} => Q-Value: {value:.3f}")
