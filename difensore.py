from random import random
import numpy as np

class difensore():
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