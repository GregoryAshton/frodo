""" Script to generate a data file with a single simulated pulse """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import frodo

# Injection parameters
pulse_injection_parameters = dict(
    C0=0.8, C1=0.1, C2=0.3, C3=0.00, C4=0.3, beta=0.003, toa=0.05,
    base_flux=0.0)

# Instantiate a flux model with 4 components
flux_model = frodo.flux_model.SinglePulseFluxModel(5)

# Generate fake data using the instantiated flux model and injection parameters
N = 1000
time = np.linspace(0, 0.1, N)
flux = flux_model(time, **pulse_injection_parameters) + np.random.normal(0, 0.1, N)
pulse_number = 0

# Write the data to a text file
df = pd.DataFrame(dict(time=time, flux=flux, pulse_number=pulse_number))
df.to_csv("fake_data.txt", index=False)

fig, ax = plt.subplots()
ax.plot(time, flux)
ax.set_xlabel("Time")
ax.set_ylabel("Flux")
fig.savefig("fake_data.png")
