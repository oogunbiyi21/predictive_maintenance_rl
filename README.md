
# **Predictive maintenance using deep reinforcement learning**

This WIP repo is essentially a fork of this one, https://github.com/golsun/deep-RL-time-series, repurposed for predictive maintenance. Deep reinforcement learing is used to find optimal strategies in keeping or replacing a piece of equipment according to its Remaining Useful Life (RUL). It uses the NASA C-MAPPS Aircraft Engine Simulator dataset

Several neural networks are compared: 
* Recurrent Neural Networks (GRU/LSTM)
* Convolutional Neural Network (CNN)
* Multi-Layer Perception (MLP)

### Dependencies

You can get all dependencies via the [Anaconda](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) environment file:

    conda env create -f env.yml

### Play with it
Just call the main function

    python main.py

