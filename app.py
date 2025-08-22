import streamlit as st
import numpy as np
from scipy.optimize import minimize, Bounds

# --- App Layout and Title ---
st.set_page_config(layout="wide")
st.title("🎯 Optimal Isocenter Calculator")
st.write("""
This app finds the optimal 3D point (isocenter) that minimizes the maximum distance
to a set of targets. It ensures that all targets are within a specified maximum distance.
""")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Parameters")
    
    # Input for the maximum allowed distance
    max_allowable_distance = st.number_input(
        "Max allowable distance from isocenter (units)", 
        min_value=1.0, 
        value=4.0, 
        step=0.5
    )
    
    # Input for the number of targets
    num_targets = st.number_input(
        "Number of targets", 
        min_value=2, 
        value=2, 
        step=1
    )
    st.markdown("---")


# --- Main Page for Target Coordinates ---
st.header("📍 Target Coordinates")

# Create columns for organized input
col1, col2, col3 = st.columns(3)
targets_list = []

for i in range(num_targets):
    with col1:
        x = st.number_input(f"Target {i+1} - X", key=f"x_{i}", value=float(i))
    with col2:
        y = st.number_input(f"Target {i+1} - Y", key=f"y_{i}", value=float(i*2))
    with col3:
        z = st.number_input(f"Target {i+1} - Z", key=f"z_{i}", value=float(15-i))
    targets_list.append([x, y, z])

# Convert list to NumPy array for calculations
targets = np.array(targets_list)

# --- Calculation and Results ---
if st.button("🚀 Calculate Optimal Isocenter", type="primary"):
    
    st.header("📈 Results")

    # --- Optimization Logic ---
    def max_distance(isocenter, targets):
        return max(np.linalg.norm(targets - isocenter, axis=1)) # np.linalg.norm -> Euclidean norm
        # max(np.linalg.norm(targets - isocenter, axis=1)) -> finding the center of the smallest possible sphere that can enclose all the target points

    initial_guess = np.mean(targets, axis=0)
    bounds = Bounds([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf])
    constraints = [{
        'type': 'ineq', 
        'fun': lambda x, i=i: max_allowable_distance - np.linalg.norm(x - targets[i])
    } for i in range(len(targets))]

    result = minimize(
        lambda x: max_distance(x, targets),
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'ftol': 1e-7, 'maxiter': 1000}
    )
    
    # --- Display Results ---
    if result.success:
        optimal_isocenter = result.x
        distances = np.linalg.norm(targets - optimal_isocenter, axis=1)

        st.subheader("Optimal Position Found")
        st.code(f"Isocenter (X, Y, Z): {np.round(optimal_isocenter, 4)}")
        
        st.subheader("Distances to Targets")
        for i, dist in enumerate(distances, start=1):
            st.markdown(f"&nbsp;&nbsp;&nbsp;**Target {i}:** `{dist:.4f}` units")

        # Final success or warning message
        if max(distances) <= (max_allowable_distance + 1e-5): # Use a small tolerance
            st.success(f"✅ SUCCESS: A valid solution was found where all distances are ≤ {max_allowable_distance} units.")
        else:
            st.warning("⚠️ WARNING: Optimization finished, but the best solution found still violates the distance constraint.")
    else:
        st.error(f"❌ FAILED: The optimization did not converge. Reason: {result.message}")