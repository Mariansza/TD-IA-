import numpy as np



int_to_char = {
    0 : 'u', # up
    1 : 'r', # right
    2 : 'd', # down
    3 : 'l' # left
} 

policy_one_step_look_ahead = {
    0 : [-1,0],  # UP: déplace de -1 en x (vers le haut)
    1 : [0,1],   # RIGHT: déplace de 1 en y (vers la droite)
    2 : [1,0],   # DOWN: déplace de 1 en x (vers le bas)
    3 : [0,-1]   # LEFT: déplace de -1 en y (vers la gauche)
}

# Cette fonction convertit une politique d'entiers en caractères pour une meilleure lisibilité
def policy_int_to_char(pi,n):
    
    pi_char = ['']

    # Boucle sur chaque cellule de la grille
    for i in range(n):
        for j in range(n):
            # Si la cellule est un état terminal (coin supérieur gauche ou coin inférieur droit)
            if i == 0 and j == 0 or i == n-1 and j == n-1:
                # On passe à la prochaine itération de la boucle
                continue

            # Ajoute le caractère correspondant à l'action de la politique à la liste pi_char
            pi_char.append(int_to_char[pi[i,j]])

    # Ajoute une chaîne vide à la fin de la liste
    pi_char.append('')

    # Convertit la liste en un tableau numpy et la redimensionne pour qu'elle ait la même forme que la grille
    return np.asarray(pi_char).reshape(n,n)



# Cette fonction évalue une politique donnée en utilisant l'algorithme d'évaluation de politique
def policy_evaluation(n, pi, v, threshhold, gamma):
    # Boucle jusqu'à ce que la différence maximale entre les anciennes et les nouvelles valeurs soit inférieure à un seuil
    while True:
        delta = 0
        for i in range(n):
            for j in range(n):
                if (i == 0 and j == 0) or (i == n-1 and j == n-1):
                    continue
                v_old = v[i, j]
                action = pi[i, j]
                # Calcule les coordonnées de la prochaine cellule en fonction de l'action
                next_i, next_j = i + policy_one_step_look_ahead[action][0], j + policy_one_step_look_ahead[action][1]
                next_i = max(0, min(n-1, next_i))  # Ensure the agent stays within the grid
                next_j = max(0, min(n-1, next_j))
                new_value = -1 + gamma * v[next_i, next_j]
                v[i, j] = new_value
                delta = max(delta, abs(v_old - new_value))
                # Si la différence maximale est inférieure au seuil, arrête la boucle
        if delta < threshhold:
            break
    return v

# Cette fonction améliore une politique donnée en utilisant l'algorithme d'amélioration de politique
def policy_improvement(n, pi, v, gamma):
    new_pi = np.zeros_like(pi)
    policy_stable = True
    for i in range(n):
        for j in range(n):
            if (i == 0 and j == 0) or (i == n-1 and j == n-1):
                continue
            old_action = pi[i, j]
            actions = [0, 1, 2, 3]
            action_values = []
            for action in actions:
                next_i, next_j = i + policy_one_step_look_ahead[action][0], j + policy_one_step_look_ahead[action][1]
                next_i = max(0, min(n-1, next_i))
                next_j = max(0, min(n-1, next_j))
                action_values.append(-1 + gamma * v[next_i, next_j])
            new_action = np.argmax(action_values)
            new_pi[i, j] = new_action
            if old_action != new_action:
                policy_stable = False
    return new_pi, policy_stable

def policy_initialization(n):
    return np.random.choice([0, 1, 2, 3], size=(n, n))


def policy_iteration(n,Gamma,threshhold):

    pi = policy_initialization(n=n)

    v = np.zeros(shape=(n,n))

    while True:

        v = policy_evaluation(n=n,v=v,pi=pi,threshhold=threshhold,gamma=Gamma)

        pi , pi_stable = policy_improvement(n=n,pi=pi,v=v,gamma=Gamma)

        if pi_stable:

            break

    return pi , v


n = 4

Gamma = [0.8,0.9,0.98]

threshhold = 1e-4

# Boucle sur chaque valeur de Gamma dans la liste
for _gamma in Gamma:
    # Appelle la fonction policy_iteration avec la taille de la grille n, la valeur courante de Gamma et le seuil
    pi , v = policy_iteration(n=n,Gamma=_gamma,threshhold=threshhold)

    # Convertit la politique d'entiers en caractères pour une meilleure lisibilité
    pi_char = policy_int_to_char(n=n,pi=pi)

    print()
    print("Gamma = ",_gamma)
    print()

    print(pi_char)

 
    print()
    print()

    print(v)