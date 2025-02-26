import random
import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, id, domain_size=10):
        self.id = id
        self.domain = list(range(domain_size))
        self.value = random.choice(self.domain)
        self.mailbox = []  # Mailbox for receiving messages
        self.constraints = {}  # Constraints with neighboring agents

    def send_message(self, neighbor):
        # Send the agent's current value to the neighbor
        neighbor.mailbox.append((self.id, self.value))

    def receive_messages(self):
        # Retrieve all messages from the mailbox
        messages = self.mailbox
        self.mailbox = []
        return messages

    def update_value(self, p, messages):
        if messages:
            # Calculate the current cost based on received messages
            current_cost = sum(self.constraints[neighbor_id] for neighbor_id, neighbor_value in messages if
                               self.value != neighbor_value)
            best_value = self.value
            best_cost = current_cost

            # Evaluate the cost for each value in the domain
            for value in self.domain:
                cost = sum(self.constraints[neighbor_id] for neighbor_id, neighbor_value in messages if
                           value != neighbor_value)
                if cost < best_cost:
                    best_value = value
                    best_cost = cost

            # Update the agent's value with a probability p
            if random.random() < p:
                self.value = best_value

def create_random_problem(num_agents=30, domain_size=10, k=0.2, seed=None):
    if seed:
        random.seed(seed)
    agents = [Agent(i, domain_size) for i in range(num_agents)]
    for agent in agents:
        # Create random neighbors for each agent based on probability k
        for neighbor in random.sample(agents, int(k * num_agents)):
            if agent != neighbor:
                cost = random.randint(0, 100)  # Discrete costs
                agent.constraints[neighbor.id] = cost
                neighbor.constraints[agent.id] = cost
    return agents

def calculate_global_cost(agents):
    total_cost = 0
    for agent in agents:
        for neighbor_id, cost in agent.constraints.items():
            if agent.value != agents[neighbor_id].value:
                total_cost += cost
    return total_cost / 2  # Each constraint is counted twice

def dsa_c(agents, p, max_iterations=100):
    global_costs = []
    for iteration in range(max_iterations):
        # Step 1: Send messages
        for agent in agents:
            for neighbor_id in agent.constraints:
                agent.send_message(agents[neighbor_id])
        # Step 2: Update values based on received messages
        for agent in agents:
            messages = agent.receive_messages()
            agent.update_value(p, messages)
        # Calculate global cost for the current iteration
        global_cost = calculate_global_cost(agents)
        global_costs.append(global_cost)
    return global_costs

def run_simulation(p_values, k, num_runs=30, max_iterations=100):
    results = {p: [] for p in p_values}
    for p in p_values:
        all_global_costs = []
        for run in range(num_runs):
            agents = create_random_problem(k=k, seed=run)  # Ensure the same seed for reproducibility
            global_costs = dsa_c(agents, p, max_iterations)
            all_global_costs.append(global_costs)
        # Calculate the average global costs over all runs
        avg_global_costs = np.mean(all_global_costs, axis=0)
        results[p] = avg_global_costs
    return results

p_values = [0.2, 0.7, 1]
k_values = [0.2, 0.7]

# Run simulations for different k values
results_k_02 = run_simulation(p_values, k=0.2)
results_k_07 = run_simulation(p_values, k=0.7)

def plot_results(results, k):
    for p, global_costs in results.items():
        plt.plot(global_costs, label=f'p={p}')
    plt.xlabel('Iterations')
    plt.ylabel('Global Cost')
    plt.title(f'Global Cost vs Iterations for k={k}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Plot results for different k values
plot_results(results_k_02, k=0.2)
plot_results(results_k_07, k=0.7)
