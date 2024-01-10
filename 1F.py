
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    print("Could not clear console and varaiables")

import matplotlib.pyplot as plt
import nengo
import numpy as np
import matplotlib.colors as mcolors

model = nengo.Network()



tau_loopback = 0.1

def A_prime(x):
    return x

def B_prime(x):
    return tau_loopback * x

def choice(x):
    if x[0] > 0.9:
        return 1
    elif x[0] < -0.9:
        return -1
    else: 
        return 0

for neuron_count, r in zip([100, 500], [1,2]):

    darker_colors = [mcolors.to_rgba('navy'), mcolors.to_rgba('darkgreen'), mcolors.to_rgba('darkred'), mcolors.to_rgba('purple')]
    lighter_colors = [mcolors.to_rgba('lightblue'), mcolors.to_rgba('lightgreen'), mcolors.to_rgba('lightcoral'), mcolors.to_rgba('violet')]
    
    for stim_strength in [0.2, 0.1, -0.1, -0.2]:
        
        def stimulus(t):
            return stim_strength
    
        with model:
            
            np.random.seed(100)
            
            with nengo.Network() as net:
                
                # Integrator taken from nengo tutorial 11. Thank you nengo.
                stim = nengo.Node(lambda t: stimulus(t))
            
                b = nengo.Ensemble(n_neurons = neuron_count, dimensions=2, radius=r)
                
                nengo.Connection(b[0], b[0], synapse=0.1)
                nengo.Connection(stim, b[0])
                nengo.Connection(b[0], b[1], function=choice)
                
                probe_integrator = nengo.Probe(
                    b[0],
                    synapse = nengo.synapses.Alpha(tau = 0.01)
                )
                
                probe_choice = nengo.Probe(
                    b[1],
                    synapse = nengo.synapses.Alpha(tau = 0.01)
                )
                
    
                
                
                with nengo.Simulator(net, progress_bar=False) as sim:
                    
                    
                    sim.run(2)
                    sim_timescale = np.linspace(0, sim.time, sim.n_steps)
            
                    f_sum = 0
                    f_des = []
                    for step in range(len(sim_timescale)):
                        t = sim_timescale[step]
                        f = stimulus(t)
                        f_sum += f * sim.dt
                        f_des.append(f_sum)
                        
                        
                
                    plt.plot(sim.trange(), sim.data[probe_choice], label='Stimulus = ' + str(stim_strength), color = darker_colors.pop())
                    plt.plot(sim.trange(), sim.data[probe_integrator], label='Integration of ' + str(stim_strength), color = lighter_colors.pop())
                    #plt.plot(sim_timescale, f_des, color = darker_colors.pop(), label="Ideal Integration of" + str(stim_strength))
    
    plt.title("Integrated Stimulus Values over time, neuron count = " + str(neuron_count))
    #plt.legend(loc='center right')
    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Output')
    plt.grid()
    plt.show()
                
            
            
        
    
        
        
        
    
    
            
    
    
    
    
    
    
    
    
    
    
    
    