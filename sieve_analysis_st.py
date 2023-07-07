# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:22:00 2023

@author: Gregor Karte
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import general_functions as gf

def percentile(p, x_cum, sieve_apertures):
    # Calculate the particle size corresponding to a given percentile
    x_l = max(x_cum[x_cum < p])
    x_h = min(x_cum[x_cum >= p])
    d_l = max(sieve_apertures[1:][x_cum < p])
    d_h = min(sieve_apertures[1:][x_cum >= p])
    d_p = (p - x_l) / (x_h - x_l) * (d_h - d_l) + d_l
    return d_p

def std(x_cum, sieve_apertures):
    # Calculate the standard deviation of the particle size distribution
    d_84 = percentile(0.84, x_cum, sieve_apertures)
    d_16 = percentile(0.16, x_cum, sieve_apertures)
    std = (d_84 - d_16) / 2
    return std

def calc_psd(sieve_apertures, m_i):
    m_tot = sum(m_i)
    x_i = m_i / m_tot
    x_cum = np.cumsum(x_i)

    mean_diameters = []
    for i in range(1, len(sieve_apertures)):
        d_mi = (sieve_apertures[i-1] + sieve_apertures[i]) / 2
        mean_diameters.append(d_mi)
    d_pi = np.array(mean_diameters)
    d_mean = round(1 / sum(x_i / d_pi))
    try:
        d_median = round(percentile(0.5, x_cum, sieve_apertures))
    except ValueError:
        d_median = -1
    try:
        std_val = std(x_cum, sieve_apertures)
    except ValueError:
        std_val = -1
    
    return x_i, x_cum, d_pi, d_mean, d_median, std_val
    
def plot_psd(inp):
    x_i, x_cum, d_pi, d_mean, d_median = inp
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), dpi=100)
    fig.tight_layout(pad=0.5)
    xmin, xmax, ymin, ymax = 0, max(sieve_apertures), 0, 1
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('$d_p\;/\;\mathrm{\mu m}$')
    ax.set_ylabel('mass fraction / -')
    ax.grid()
    ax.grid(which='minor', color='gray', linestyle=':', linewidth=0.5)
    ax.minorticks_on()
    ax.plot(sieve_apertures[1:], x_cum, "x-", color="black", linewidth=1, label="Cumulative PSD")
    ax.plot(d_pi, x_i, ".-", color="black", linewidth=1, label="PSD")
    ax.plot([d_mean, d_mean], [0, 1], "--", color="black", linewidth=0.8, label=f"Harmonic mean diameter = {d_mean}")
    ax.plot([d_median, d_median], [0, 1], "-.", color="black", linewidth=0.8, label=f"Median diameter = {d_median}")
    ax.legend()

    return fig

# Streamlit app
st.title("Particle Size Distribution from Sieve Analysis")
st.markdown("Enter the sieve apertures and masses per sieve below:")

# User input for sieve apertures
sieve_apertures_input = st.text_input("Sieve Apertures incl. 0 and fictive max. particle size (comma-separated)", "0, 125, 150, 180, 210, 250, 300, 350, 420, 500, 600")
sieve_apertures = np.array([int(x.strip()) for x in sieve_apertures_input.split(",")])

mass_fractions_input = st.text_input("Masses per sieve (comma-separated)", "0., 4.725, 17.85, 18.75, 20.775, 31.2, 21.675, 16.875, 17.4, 0.75")
m_i = np.array([float(x.strip()) for x in mass_fractions_input.split(",")])

if len(m_i) != len(sieve_apertures[1:]):
    st.warning("number of masses must be one less than number of sieve apertures")
else:
    # calc
    res = calc_psd(sieve_apertures, m_i)
    x_i, x_cum, d_pi, d_mean, d_median, std_val = res
    # Generate plot
    fig = plot_psd(res[:-1])
    # Create pandas DataFrame
    intervals = np.diff(sieve_apertures)

    data = {
        "d_min": sieve_apertures[:-1],
        "d_max": sieve_apertures[1:],
        "Interval": intervals,
        "Mean Diameter": d_pi,
        "Mass": m_i,
        "Mass Fraction": x_i,
        "Cumulative Mass Fraction": x_cum
    }
    df = pd.DataFrame(data)

    
    # checks
    st.write(f"total mass = {round(sum(m_i),2)}")
    # Display plot
    st.pyplot(fig)
    # Display DataFrame
    st.write(df)
    


    # Dataframe and figure to xlsx + download button
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine = 'xlsxwriter')
    df.to_excel(writer, sheet_name="Sieve analysis data", float_format="%.5f", startrow=0)
    writer.sheets["Sieve analysis data"].insert_image('J1', "", {'image_data': gf.fig2img(fig)})
    writer.close()
    filename = f'SieveAnalysisResult_{gf.str_date_time()}'
    st.download_button(
        label="Download as xlsx-file",
        data=output.getvalue(),
        file_name= f'{filename}.xlsx'
    )


