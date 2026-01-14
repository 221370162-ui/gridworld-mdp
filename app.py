import streamlit as st
import matplotlib.pyplot as plt
from mdp import GridWorldMDP, value_iteration, policy_iteration

st.title("Grid-World MDP Visualization")

gamma = st.slider("Discount Factor (γ)", 0.1, 0.99, 0.9)

algorithm = st.selectbox("Algorithm", ["Value Iteration", "Policy Iteration"])

mdp = GridWorldMDP(
    grid=(5, 5),
    terminals={(4, 4): 10, (4, 0): -10},
    obstacles={(1, 1), (2, 2)},
    gamma=gamma
)

if st.button("Run"):
    if algorithm == "Value Iteration":
        V, policy, history = value_iteration(mdp)
        st.success(f"Converged in {len(history)} iterations")
    else:
        V, policy, history = policy_iteration(mdp)
        st.success(f"Converged in {len(history)} iterations")

    fig, ax = plt.subplots()
    for s, v in V.items():
        ax.text(s[1], mdp.rows - s[0] - 1, f"{v:.2f}", ha="center", va="center")
    ax.set_title("Value Function")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    for s, a in policy.items():
        if a:
            ax.text(s[1], mdp.rows - s[0] - 1, a, ha="center", va="center")
    ax.set_title("Optimal Policy")
    st.pyplot(fig)
