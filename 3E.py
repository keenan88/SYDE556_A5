
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    print("Could not clear console and varaiables")

import matplotlib.pyplot as plt
import nengo
import numpy as np

def initialization(x):
    return 0

sim_len_seconds = 100
sublist_len = 2000
N_sublists = int(sim_len_seconds * 1000 / sublist_len)

N_simulations = 10
rmses = np.zeros((N_simulations, N_sublists))

for sim_idx in range(N_simulations):
    print(sim_idx)

    model = nengo.Network()
    
    with model:
        
        np.random.seed(sim_idx*100 + 1)
        
        with nengo.Network() as net:
            
            pre = nengo.Ensemble(n_neurons=200, dimensions=1)
            post = nengo.Ensemble(n_neurons=200, dimensions=1)
            error = nengo.Ensemble(n_neurons=200, dimensions=1)
            stim = nengo.Node(nengo.processes.WhiteSignal(period=100, high=2, rms=0.3))
            
            #difference = nengo.Node(size_in=1)
            nengo.Connection(pre, error, transform=-1)
            nengo.Connection(post, error, transform=1)
            #nengo.Connection(difference, error)
            
            # Learning connection, calculates difference between post's
            # output value, and the desired output value given be pre
            c = nengo.Connection(
                pre, 
                post, 
                function = initialization, 
                learning_rule_type = nengo.PES(learning_rate = 1e-5)
            )
            
            nengo.Connection(stim, pre)
            nengo.Connection(error, c.learning_rule)
            
            pre_probe = nengo.Probe(
                pre,
                synapse = nengo.synapses.Alpha(tau = 0.01)
            )
            
            post_probe = nengo.Probe(
                post,
                synapse = nengo.synapses.Alpha(tau = 0.01)
            )
            
            error_probe = nengo.Probe(
                error,
                synapse = nengo.synapses.Alpha(tau = 0.01)
            )
            
            stim_probe = nengo.Probe(
                stim,
                synapse = nengo.synapses.Alpha(tau = 0.01)
            )
            
            
            with nengo.Simulator(net, progress_bar=False) as sim:
                
                sim.run(sim_len_seconds)
                    
                #plt.plot(sim.trange(), sim.data[stim_probe], label = "Stimulus")
                #plt.plot(sim.trange(), sim.data[pre_probe], label = "Pre")
                #plt.plot(sim.trange(), sim.data[post_probe], label = "Post")
                #plt.plot(sim.trange(), sim.data[error_probe], label = "Error")
    
        for slice_idx in range(N_sublists):
            start_idx = slice_idx * sublist_len
            end_idx = (slice_idx + 1) * sublist_len
            stim_slice = sim.data[stim_probe][start_idx : end_idx]
            post_slice = sim.data[post_probe][start_idx : end_idx]
            
            rmse = np.square(np.mean(np.square(stim_slice - post_slice)))
            rmses[sim_idx, slice_idx] = rmse
    
rmse_avgs = np.mean(rmses, axis=0)
    
rmse_times = np.arange(2, sim_len_seconds + 1, 2)
plt.scatter(rmse_times, rmse_avgs)
    
plt.title("RMSE over 2 second time slices, averaged over 10 simulations")
plt.xlabel('Time (s)')
plt.ylabel('RMSE')
plt.grid()
plt.show()
                
            
            
        
    
        
        
        
    
    
            
    
    
    
    
    
    
    
    
    
    
    
    