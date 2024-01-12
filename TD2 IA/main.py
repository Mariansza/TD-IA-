import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time

# Cette fonction met à jour la table Q en utilisant l'algorithme Q-Learning.
def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    # Récupère la valeur Q actuelle pour l'état s et l'action a.
    current_q_value = Q[s][a]
    # Récupère la valeur Q maximale pour l'état suivant sprime.
    max_future_q_value = max(Q[sprime])
    # Calcule la nouvelle valeur Q en utilisant la formule du Q-Learning.
    new_q_value = (1 - alpha) * current_q_value + alpha * (r + gamma * max_future_q_value)
    # Met à jour la valeur Q pour l'état s et l'action a.
    Q[s][a] = new_q_value
    # Retourne la table Q mise à jour.
    return Q



# Cette fonction implémente la politique epsilon-greedy pour la sélection d'actions.
def epsilon_greedy(Q, s, epsilon):
    # Si un nombre aléatoire est inférieur à epsilon, alors on choisit une action au hasard (exploration).
    if random.uniform(0, 1) < epsilon:
        action = random.choice(range(len(Q[s])))
    else:
        # Sinon, on choisit l'action qui a la plus grande valeur Q (exploitation).
        action = np.argmax(Q[s])
    return action

if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode="human")

    env.reset()
    state = env.s
    env.render()

    # Définir l'ensemble des États
    states = [(taxi_x, taxi_y, passenger_loc, destination_loc)
              for taxi_x in range(5)
              for taxi_y in range(5)
              for passenger_loc in range(5)  
              for destination_loc in range(5)  
              ]

    # Initialise la table Q avec des zéros.
    Q = np.zeros((len(states), env.action_space.n))

    # Définit les paramètres de l'algorithme Q-Learning.
    alpha = 0.05
    gamma = 0.9
    epsilon = 0.1 

    n_epochs = 50 
    max_itr_per_epoch = 200 
    rewards = []

    # Exécute l'algorithme Q-Learning pour le nombre spécifié d'épisodes.
    for e in range(n_epochs):
        r = 0

        env.reset()
        state = env.s

    # Exécute l'algorithme Q-Learning pour le nombre maximum d'itérations.
        for _ in range(max_itr_per_epoch):
            action = epsilon_greedy(Q=Q, s=state, epsilon=epsilon)

            next_state, reward, done, _, info = env.step(action)

            r += reward

            next_s = next_state  

            Q = update_q_table(
                Q=Q, s=state, a=action, r=reward, sprime=next_s, alpha=alpha, gamma=gamma
            )

            state = next_s

            if done:
                break

        print("Épisode #", e, " : r = ", r)

        rewards.append(r)

    print("Récompense moyenne = ", np.mean(rewards))

    print("Entraînement terminé.\n")


    env.close()