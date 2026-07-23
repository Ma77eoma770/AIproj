import networkx as nx

from grafo import graph
from attaccante import attaccante
from difensore import difensore

class environment():
    def __init__(self,pathlength):
        self.G = graph(pathlength)
        self.reset()
    
    def reset(self):
        self.G.reset()
        self.sim_on = True
        self.att = attaccante(self.G.start_node, self.G.G.nodes[self.G.start_node]['heuristic'])
        if not hasattr(self, 'dif') or self.dif is None:
            self.dif = difensore()
        
        # Stato POMDP tracciato dall'ambiente
        self.distanceFromStart = 0
        self.distanceToEnd = self.G.pathlength - 1
        self.extimatedPatience = 100.0

        self.patience_history = {self.G.start_node: self.extimatedPatience}

        self.stepped_traps = 0
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
            reward -= 2.5
        
        # 3. L'attaccante risponde muovendosi sul grafo modificato
        att_status = self.att.step(self)
        if att_status[0] == 'Stuck':
            self.sim_on = False
            print("L'attaccante è rimasto bloccato senza alternative. Simulazione terminata.")  

        undirected_G = self.G.G.to_undirected(as_view=True)

        # Aggiornamento dello stato osservabile
        if att_status[0] in ('Normal Step', 'Jumped'):
            if att_status[0] == 'Jumped':
                # Restore to the parent's snapshot patience and apply jump penalty
                parent_node = att_status[3]
                self.extimatedPatience = self.patience_history.get(parent_node, self.extimatedPatience)
                self.extimatedPatience = max(0, self.extimatedPatience - (5 + (self.G.G.nodes[att_status[1]]['heuristic'] - self.G.G.nodes[att_status[2]]['heuristic'])))
            else:
                # Normal step: gain 2 patience CHECK ON 100 HARDCODED !!!!!!!!!!!!!!!!!!!!!!
                self.extimatedPatience = min(100, self.extimatedPatience)
                
            if self.att.current_node not in self.patience_history:
                self.patience_history[self.att.current_node] = self.extimatedPatience
            try:
                self.distanceToEnd = nx.shortest_path_length(undirected_G, source=self.att.current_node, target=self.G.target_node)
            except nx.NetworkXNoPath:
                self.distanceToEnd = 1000

            # Instant reward
            stepped_traps = sum(1 for n in self.att.explored_nodes if self.G.G.nodes[n].get('is_bait', False))
            if stepped_traps > self.stepped_traps:
                reward += 5
                self.stepped_traps = stepped_traps 

        next_state = self.getCurrentState()
        done = False

        # 4. Condizioni di terminazione e ricompense finali
        if att_status[0] == 'Target Reached':
            # Vittoria: ha munto l'attaccante fino alla fine. Diamo il maxi-reward!
            shortest_path_length = nx.shortest_path_length(self.G.G, source=self.G.start_node, target=self.G.target_node)
            reward += 5.0 * (len(self.att.explored_nodes) - shortest_path_length + 1)
            done = True
            self.sim_on = False
            
        elif att_status[0] == 'GaveUp':
            # Fallimento: ha esagerato e l'attaccante si è stufato (NIENTE punti esplorazione)
            reward -= 50.0
            done = True
            self.sim_on = False
            
        elif att_status[0] == 'Stuck':
            # L'attaccante si è incastrato
            reward -= 20.0
            done = True
            self.sim_on = False

        return prev_state, dif_action, reward, next_state, done
    
    def getCurrentState(self):
            # 1. Bucket della pazienza (0-5) - Usiamo la pazienza stimata (POMDP)
            patience_bucket = int(self.extimatedPatience // 20)
            
            # 2. Controllo topologia locale: l'attaccante è già su una trappola?
            current_node_data = self.G.G.nodes[self.att.current_node]
            is_on_trap = 1 if current_node_data.get('is_bait', False) else 0
            
            # 3. Rileviamo se ci sono trappole adiacenti collegate al nodo corrente
            has_bait = 0
            has_loop = 0
            has_deathend = 0
            for neighbor in self.G.G.successors(self.att.current_node):
                neighbor_data = self.G.G.nodes[neighbor]
                if neighbor_data.get('is_bait', False):
                    name = neighbor_data.get('name', '')
                    if '_loop' in name:
                        has_loop = 1
                    elif '_deathend' in name:
                        has_deathend = 1
                    elif '_bait' in name:
                        has_bait = 1
            
            # 4. Restituiamo lo stato esteso e compatto
            return (
                self.distanceToEnd, 
                patience_bucket,
            )
    
    def get_neighbors(self,current_node):
        return self.G.get_neighbors(current_node)
    
    def checkTarget(self, node):
        return node == self.G.target_node