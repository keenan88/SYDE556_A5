
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



model = nengo.Network()

with model:
    
    np.random.seed(100)
    
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
            learning_rule_type = nengo.PES(learning_rate = 1e-4)
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
            
            
            sim.run(10)
                
            #plt.plot(sim.trange(), sim.data[stim_probe], label = "Stimulus")
            plt.plot(sim.trange(), sim.data[pre_probe], label = "Pre")
            plt.plot(sim.trange(), sim.data[post_probe], label = "Post")
            #plt.plot(sim.trange(), sim.data[error_probe], label = "Error")

plt.title("PES Learning Rule Approximation of identity matrix")
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Neuron Output')
plt.grid()
plt.show()
                
            
            
        
    
        
        
        
    
    
            
    
    
    
    
    
    
    
    
    
    
    
    