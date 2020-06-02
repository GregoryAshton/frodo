"""
Script to convert an AR file to a CSV file

Usage:

    $ python convert_art.py file.ar

"""
import argparse
import sys
import subprocess
import os

import numpy as np
import psrchive
import pandas as pd

filename = sys.argv[1]
filename_cent = filename.replace(".ar", ".centered")
arch = psrchive.Archive_load(filename)
npulses = arch.get_data().shape[0]
arch.centre()
arch.remove_baseline()
arch.set_filename(filename_cent)
arch.unload()

times = []
fluxes = []
pulse_number = []

npulses = 3

for pn in range(npulses):
    cmd = 'pam {} -x "{} {}" -e sar{} -F -p'.format(
        filename_cent, pn, pn, pn)
    print("Calling: {}".format(cmd))
    os.system(cmd)
    subint_filename = filename.replace(".ar", ".sar{}".format(pn))

    arch = psrchive.Archive_load(subint_filename)
    flux = np.squeeze(arch.get_data())

    start_time = arch.start_time().in_days()
    end_time = arch.end_time().in_days()
    time_array = np.linspace(start_time, end_time, len(flux))

    times += list(time_array)
    fluxes += list(flux)
    pulse_number += [pn] * len(time_array)

os.system("rm *.sar*")

df = pd.DataFrame(dict(time=times, flux=fluxes, pulse_number=pulse_number))

# Write the data to a CSV file
df.to_csv(
    filename.replace(".ar", ".txt"),
    columns=["time", "flux", "pulse_number"],
    header=True, index=False, float_format=np.float64
)
