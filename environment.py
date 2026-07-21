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
        self.dif = difensore()
        
        # Stato POMDP tracciato dall'ambiente
        self.distanceFromStart = 0
        self.distanceToEnd = self.G.pathlength - 1
        self.extimatedPatience = 100.0

        self.patience_history = {self.G.start_node: self.extimatedPatience}
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

        undirected_G = self.G.G.to_undirected()

        # Aggiornamento dello stato osservabile
        if att_status[0] == 'Normal Step':
            if self.G.G.nodes[att_status[1]].get('is_bait', False):
                self.extimatedPatience = max(0, self.extimatedPatience - 15)
            else:
                self.extimatedPatience = min(100, self.extimatedPatience + 5)
            self.patience_history[att_status[1]] = self.extimatedPatience
            try:
                self.distanceToEnd = nx.shortest_path_length(undirected_G, source=self.att.current_node, target=self.G.target_node)
            except nx.NetworkXNoPath:
                self.distanceToEnd = 1000
        
        if att_status[0] == 'Backtracked':
            self.extimatedPatience = self.patience_history.get(att_status[1], self.extimatedPatience)
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
            # 1. Bucket della pazienza (0-5) -> GIA' OTTIMO
            patience_bucket = int(self.extimatedPatience // 20)
            
            # 2. Controllo topologia locale: l'attaccante è già su una trappola?
            current_node_data = self.G.G.nodes[self.att.current_node]
            is_on_trap = 1 if current_node_data.get('is_bait', False) else 0
            
            # 3. Restituiamo lo stato compatto
            return (
                self.distanceToEnd, 
                patience_bucket,
                is_on_trap
            )
    
    def get_neighbors(self,current_node):
        return self.G.get_neighbors(current_node)
    
    def checkTarget(self, node):
        return node == self.G.target_node