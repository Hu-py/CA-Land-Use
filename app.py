import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List
from scipy.ndimage import uniform_filter, convolve, distance_transform_edt
import imageio
import pandas as pd
import time

# -------------------------------
# Constants
# -------------------------------
CLASSES = {0: 'Vacant', 1: 'Urban', 2: 'Agriculture', 3: 'Forest'}
PALETTE = {0: (0.85,0.85,0.85), 1:(0.85,0.15,0.15), 2:(0.95,0.85,0.25), 3:(0.2,0.6,0.3)}
BASE_P = np.array([
    [0.00, 0.25, 0.40, 0.35],
    [0.02, 0.90, 0.05, 0.03],
    [0.02, 0.20, 0.70, 0.08],
    [0.01, 0.10, 0.09, 0.80],
], dtype=float)
ALLOWED = (BASE_P>0).astype(float)

# -------------------------------
# Data classes
# -------------------------------
@dataclass
class CAState:
    grid: np.ndarray
    suitability: Dict[int, np.ndarray]
    access: np.ndarray
    history_shares: List[np.ndarray]
    frames: List[np.ndarray]

# -------------------------------
# Utilities
# -------------------------------
def make_base_layers(n=100, m=100, seed=1):
    rng = np.random.default_rng(seed)
    # simple initialization
    grid = np.zeros((n,m), dtype=int)
    for _ in range(5):
        ci,cj = rng.integers(n//3,2*n//3), rng.integers(m//3,2*m//3)
        rr = rng.integers(3,7)
        ii,jj = np.ogrid[:n,:m]
        mask = (ii-ci)**2 + (jj-cj)**2 <= rr**2
        grid[mask] = 1
    grid[rng.random((n,m))<0.12]=2
    grid[rng.random((n,m))<0.2]=3

    suitability = {}
    for k in CLASSES.keys():
        s = uniform_filter(rng.random((n,m)), size=7)
        s = (s - s.min())/(s.max()-s.min()+1e-9)
        suitability[k] = s

    # roads and access
    roads = np.zeros((n,m), dtype=np.uint8)
    center = (n//2, m//2)
    for ang in np.linspace(0,2*np.pi,6,endpoint=False):
        for r in range(min(n,m)//2):
            i=int(center[0]+r*np.sin(ang)); j=int(center[1]+r*np.cos(ang))
            if 0<=i<n and 0<=j<m: roads[i,j]=1
    dist = distance_transform_edt(1-roads)
    access = 1.0/(1.0+dist)
    access = (access-access.min())/(access.max()-access.min()+1e-9)
    return grid, suitability, access

def render_grid(grid):
    rgb = np.zeros((grid.shape[0], grid.shape[1],3),dtype=float)
    for k,c in PALETTE.items():
        rgb[grid==k]=c
    return rgb

def neighborhood_share(grid,target_class,radius=1):
    k = 2*radius+1
    mask = (grid==target_class).astype(float)
    num = convolve(mask,np.ones((k,k)),mode='nearest')
    den = convolve(np.ones_like(mask),np.ones((k,k)),mode='nearest')
    num -= mask; den -= 1.0
    den = np.maximum(den,1.0)
    return num/den

def softmax(x,temp=1.0,axis=0):
    x = x/float(max(temp,1e-6))
    x = x - np.max(x,axis=axis,keepdims=True)
    e = np.exp(x)
    return e/np.sum(e,axis=axis,keepdims=True)

# -------------------------------
# CA Step
# -------------------------------
def step_ca(state:CAState, weights:Dict[str,float], radius:int, temp:float):
    grid = state.grid
    n,m = grid.shape
    K=len(CLASSES)
    score = np.zeros((K,n,m))
    neigh = {k: neighborhood_share(grid,k,radius) for k in CLASSES}
    for k in CLASSES.keys():
        s = weights['wS']*state.suitability[k] + weights['wN']*neigh[k] + weights['wA']*state.access + weights['wI']*(grid==k)
        score[k] = s
    base = BASE_P[grid]
    allow = ALLOWED[grid]
    S = score.reshape(K,-1)
    prior = (base*allow).reshape(-1,K).T + 1e-6
    logits = S + np.log(prior)
    probs = softmax(logits,temp=temp,axis=0)
    rnd = np.random.default_rng().random(probs.shape[1])
    cum = np.cumsum(probs,axis=0)
    choice = (cum<rnd).sum(axis=0).reshape(n,m)
    state.grid = choice
    state.history_shares.append(np.bincount(choice.ravel(),minlength=K)/choice.size)
    state.frames.append((render_grid(choice)*255).astype(np.uint8))
    return state

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("CA Land-Use Change Simulation")

n = st.sidebar.slider("Rows",40,200,100)
m = st.sidebar.slider("Cols",40,200,100)
steps = st.sidebar.slider("Steps",1,100,20)
seed = st.sidebar.number_input("Seed",0,9999,1)

# Weights sliders
wS = st.sidebar.slider("wS Suitability",0.0,3.0,0.5)
wN = st.sidebar.slider("wN Neighborhood",0.0,3.0,2.0)
wA = st.sidebar.slider("wA Access",0.0,3.0,0.2)
wI = st.sidebar.slider("wI Inertia",0.0,3.0,0.2)
temp = st.sidebar.slider("Temperature",0.1,2.0,0.6)
speed = st.sidebar.slider("🎞️ Animation speed (s per frame)",0.01,1.0,0.1)

export_gif = st.sidebar.checkbox("Export GIF", value=False)
export_csv = st.sidebar.checkbox("Export CSV", value=True)

if st.button("Run Simulation"):
    rng = np.random.default_rng(seed)
    grid, suitability, access = make_base_layers(n,m,seed)
    state = CAState(grid, suitability, access, history_shares=[np.bincount(grid.ravel(),minlength=len(CLASSES))/grid.size],
                    frames=[(render_grid(grid)*255).astype(np.uint8)])
    for t in range(steps):
        state = step_ca(state,weights={'wS':wS,'wN':wN,'wA':wA,'wI':wI,'wD':0.0}, radius=2, temp=temp)
        
    # Animation (all frames)
    st.subheader("Dynamic Animation")
    gif_path = "temp_ca.gif"
    imageio.mimsave(gif_path,state.frames,duration=speed)
    st.image(gif_path, use_column_width=True)
    
    # Initial vs final
    st.subheader("Initial / Final States")
    col1,col2=st.columns(2)
    with col1: st.image(state.frames[0], caption="Initial", use_column_width=True)
    with col2: st.image(state.frames[-1], caption="Final", use_column_width=True)



    # Plot class shares over time
    st.subheader("Class Shares Over Time")
    fig,ax=plt.subplots()
    hist = np.array(state.history_shares)
    for k,name in CLASSES.items():
        ax.plot(hist[:,k],label=name)
    ax.set_xlabel("Step"); ax.set_ylabel("Share"); ax.set_ylim(0,1)
    ax.legend(); st.pyplot(fig)

    # Export CSV
    if export_csv:
        df = pd.DataFrame(hist, columns=[CLASSES[k] for k in range(len(CLASSES))])
        df.insert(0,'step',np.arange(len(hist)))
        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        path_csv=f"ca_shares_{ts}.csv"
        df.to_csv(path_csv,index=False)
        st.write("Saved CSV:",path_csv)
