
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    print("Could not clear console and varaiables")

import matplotlib.pyplot as plt
import nengo
import numpy as np
import csv

model = nengo.Network()

def A_B(q, theta):
    A = np.zeros((q, q))
    B = np.zeros((q, 1))
    for i in range(q):
        B[i] = (-1.)**i * (2*i+1)
        for j in range(q):
            A[i,j] = (2*i+1)*(-1 if i<j else (-1.)**(i-j+1)) 
    A = A / theta
    B = B / theta
    
    return A, B
    
q = 6
theta = 0.5
A, B = A_B(q, theta)

tau_loopback = 0.1

def A_prime(x):
    #print(x)
    
    transformer = np.matrix(tau_loopback * A + np.identity(q))
    
    #print(transformer)
    #print(type(transformer))
    #print(transformer.shape)
    
    transformed = transformer * np.matrix(x).T
    transformed = transformed.T
    transformed = transformed.tolist()[0]
    
    #print(type(transformed))
    #print(transformed)
    #print(transformed.shape)
    
    #print()
    return transformed

def B_prime(x):
    return (tau_loopback * B * x).T


sim_length_s = 4
sim_dt_s = 0.001    
N_samples = int(sim_length_s / sim_dt_s)

with model:
    
    np.random.seed(1000)
    
    with nengo.Network() as net:
        
        N1 = 1000
        
        radius_ = 1
            
        stim = nengo.Node(lambda t: np.sin(2*np.pi*t*2) if t<2 else np.sin(2*np.pi*t))
        
        ens1 = nengo.Ensemble(
            radius = radius_,
            n_neurons = N1, 
            dimensions = q
        )
        
        
        
        choice_ens = nengo.Ensemble(
            radius = radius_,
            n_neurons = 50, 
            dimensions = 1
        )
        
        
        stim_connection = nengo.Connection(
            stim, 
            ens1,
            synapse = 0.1,
            #transform = B,
            function = B_prime
        )
        
        
        loopback = nengo.Connection(
            ens1, 
            ens1,
            synapse = 0.1,
            #transform = A,
            function = A_prime
        )
        
        target = np.concatenate(
            (np.zeros(int(N_samples/2)) + 1, 
            np.zeros(int(N_samples/2)) - 1),
            axis = 0
        )
        target = np.matrix(target).T
        
        evaluation_points = []
        
        save_file = 'Q2_curves.csv'
        with open(save_file, 'r') as file:
            csv_reader = csv.reader(file)
        
            for row in csv_reader:
                evaluation_points.append(row)
        
        evaluation_points = np.matrix(evaluation_points)
        
        choice_conn = nengo.Connection(
            ens1, 
            choice_ens,
            eval_points = evaluation_points,
            function = target
        )
        
        probe_stim = nengo.Probe(
            target = stim, 
            synapse = nengo.synapses.Alpha(tau = 0.01)
        ) 
        
        probe_ens1 = nengo.Probe(
            ens1,
            synapse = nengo.synapses.Alpha(tau = 0.01)
        )
        
        probe_choice = nengo.Probe(
            choice_ens,
            synapse = nengo.synapses.Alpha(tau = 0.01)
        )
        
        
        
        with nengo.Simulator(net, progress_bar=False) as sim:
            
            
            sim.run(sim_length_s)
                
            #plt.plot(sim.trange(), sim.data[probe_ens1], label="Ensemble1")
            plt.plot(sim.trange(), sim.data[probe_stim], label="stimulus")
            plt.plot(sim.trange(), sim.data[probe_choice], label="Choice")
            

            
            

plt.title("Integrated Stimulus Values over time")
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Neuron Output')
plt.grid()
plt.show()
                
            
            
        
    
        
        
        
    
    
            
    
    
    
    
    
    
    
    
    
    
    
    