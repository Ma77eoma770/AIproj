from attaccante import attaccante
from difensore import difensore
from environment import environment


import random as rd

# 1. Inizializziamo il difensore globale che manterrà la memoria tra i vari grafi
difensore_in_training = difensore(action_dimension=4, learning_rate=0.1, epsilon=0.9, gamma=0.95,epsilon_decay=0.9997, min_epsilon=0.01)

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
env_test.G.save_graph_png(f"simulazione_step_{step_sim}_inizio.png", attacker_node=env_test.att.current_node, attacker_patience=env_test.att.patience)

while not done and env_test.sim_on:
    step_sim += 1
    print(f"\n--- TURNO {step_sim} ---")
    
    # Rileviamo la posizione attuale dell'attaccante
    att_node = env_test.att.current_node
    node_name = env_test.G.G.nodes[att_node]['name']
    print(f"📍 Posizione Attaccante: Nodo {att_node} [{node_name}]")
    
    # Rileviamo lo stato che viene passato alla rete
    curr_state = env_test.getCurrentState()
    print(f"📊 Stato POMDP (visto dal difensore): Distanza_Start={curr_state[0]}, Distanza_Target={curr_state[1]}, Bucket_Pazienza_Stimata={curr_state[2]}")
    
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
    env_test.G.save_graph_png(nome_foto, attacker_node=env_test.att.current_node, attacker_patience=env_test.att.patience)
    # La funzione save_graph_png ha già un suo print integrato[cite: 3]
    
print("\n" + "="*50)
print("SIMULAZIONE TERMINATA")
print("="*50)
print(f"🏆 Reward Totale accumulato: {total_test_reward}")
print(f"🔍 Nodi totali esplorati (e sprecati) dall'attaccante: {len(env_test.att.explored_nodes)}")
print(f"Q-Table finale del difensore: {difensore_in_training.q_table}")