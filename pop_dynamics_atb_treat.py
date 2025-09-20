
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the differential equations (Equations 10-15 from the document)
def population_dynamics(y, t, C, Vmax, Vmin, VmaxP, VminP, VmaxB, VminB, K, e, km, 
                        Mmin, Mmax, pd, pr, ka, f, x, fNP, fPN, fNB, fBN, w, wB, 
                        di, dd, Amax, Dose, initial_dose_time):
    
    # Unpack the current state variables
    R, N, D, P, B, A = y

    # Ensure concentrations and populations are non-negative
    R = max(0, R)
    N = max(0, N)
    D = max(0, D)
    P = max(0, P)
    B = max(0, B)
    A = max(0, A)

    # Total viable cell density
    NT = N + P + B

    # Pharmacodynamic functions (Equations 3-5 from the document)
    # Ensure R+km is not zero to avoid division by zero
    if (R + km) == 0:
        psi_max_R = 0
        psi_min_R = Vmin * (1 - pr)
    else:
        psi_max_R = Vmax * (R / (R + km))
        psi_min_R = Vmin * (1 - pr) + pr * Vmin * (R / (R + km))
    
    # Ensure (N+ka) is not zero
    if (N + ka) == 0:
        M_N = Mmin
    else:
        M_N = Mmin + pd * (Mmax * (N / (N + ka)))

    # H(A, N, R) function (part of Eq. 1 & 2)
    # Avoid division by zero if A/M_N or psi_max_R are problematic
    if M_N == 0:
        H_ANR = 0 # Or handle as per specific model assumptions if M_N can be 0
    elif (A / M_N) == 0:
        H_ANR = 0
    else:
        numerator = psi_max_R - psi_min_R * ((A / M_N) ** K)
        denominator = (((A / M_N) ** K) - (psi_max_R / psi_min_R))
        if denominator == 0: # Avoid division by zero
            H_ANR = 0
        else:
            H_ANR = numerator / denominator

    # theta(A, N, R) (Equation 1)
    theta_ANR = psi_max_R - H_ANR
    
    # theta_P(A, P, R) and H_P(A, P, R) for persisters
    # Assuming persister and wall populations have their own simplified PD, 
    # independent of N and R for simplicity as per document
    # "For the persister and wall populations, we assume that the Hill function is density and resource concentration independent"
    # The document states VmaxP=0.001 and VminP=-0.001, indicating very slow growth/kill.
    # For a full model, one might use a simplified Hill function without R and N dependence for P and B.
    # Here, for simplicity, we'll assume a constant growth/kill rate based on VmaxP/VminP
    # and that their H function is directly related to their kill rate if A is present.
    # The document states psi_min(R)=V_min when pr=0. So if pr=0 for P and B, then it's V_minP
    
    # Let's simplify H_P and H_B based on their V_minP/V_minB for killing effect
    # The document implies these are less susceptible and grow/kill very slowly.
    # If A is present, we assume a killing effect proportional to A, but capped by VminP/VminB.
    # The equations (12-14) use H_P(A,P,R) and H_B(A,B,R) which suggests they follow a similar Hill form.
    # Given the text "The antibiotic specific growth and sensitivity parameters of the P and B populations may differ from those of the N population; 
    # in particular, the P and B populations would have a lower maximum growth rate (i.e., VmaxP and VmaxB<Vmax) and, for a bactericidal antibiotic, 
    # a lower minimum growth rate (lower kill rate) (i.e., VminP and VminB>Vmin).",
    # and "For the persister and wall populations, we assume that the Hill function is density and resource concentration independent"
    # We will use simplified PD functions for P and B, assuming their MICs are much higher or their kill rates are very low.
    
    # For simplicity, we can define a constant kill rate for P and B if A > 0.
    # A more rigorous implementation would involve defining specific Hill functions for P and B,
    # potentially with their own M_P, M_B, K_P, K_B etc.
    
    # Let's use the given VmaxP, VminP, VmaxB, VminB directly for theta and H.
    # If A is present, they are killed at VminP/VminB, otherwise they grow at VmaxP/VmaxB.
    
    # Simplified theta for P and B (assuming antibiotic always present for killing effect when A > 0)
    theta_P_val = VminP if A > 0.1 else VmaxP # If A is effectively present, kill at VminP, else grow at VmaxP
    theta_B_val = VminB if A > 0.1 else VmaxB # Using 0.1 as a small threshold for antibiotic presence

    # H functions for D calculation (rate of death for P and B populations)
    # H represents the 'kill' part. If A is present, and VminP/VminB are negative, then H is effectively -VminP/B
    H_P_val = -theta_P_val if A > 0.1 and VminP < 0 else 0
    H_B_val = -theta_B_val if A > 0.1 and VminB < 0 else 0
    
    # If VminP/B are positive, then they are growing, and H is 0 or represents something else.
    # Given VminP/B are negative (kill rates), H_P and H_B should reflect this.
    # The document states H(A,N,R) is defined as psi_max - theta. So H_P should be psi_maxP - theta_P.
    # Let's assume psi_maxP is VmaxP and psi_minP is VminP for these populations, and they have constant MIC (Mmin).
    # This part of the model for P and B is less explicit in the text.
    # For the sake of completing the script, let's assume a simpler kill mechanism for P and B when A > 0.
    
    # For simplicity, if A is present and a population is killed (Vmin < 0), H is the absolute value of the kill rate.
    H_P = abs(VminP) if A > 0.1 and VminP < 0 else 0
    H_B = abs(VminB) if A > 0.1 and VminB < 0 else 0

    # Delta(NT) function (Equation 6)
    delta_NT = di + dd * NT

    # Differential Equations
    # dR/dt (Equation 10)
    # Ensure positive arguments for R, N, P, B.
    # Make sure to handle potential division by zero in (R+km) when calculating terms within it.
    term_R_N = 0 if (R + km) == 0 else (R / (R + km)) * N * Vmax
    term_R_P = 0 if (R + km) == 0 else (R / (R + km)) * P * VmaxP
    term_R_B = 0 if (R + km) == 0 else (R / (R + km)) * B * VmaxB

    dRdt = w * (C - R) - e * (term_R_N + term_R_P + term_R_B) + f * D * x

    # dN/dt (Equation 11)
    dNdt = theta_ANR * N - w * N - fNP * N + fPN * P - fNB * N + fBN

    # Placeholder ODEs for other compartments (D, P, B, A)
    # These should be replaced with actual model equations as needed
    dDdt = 0
    dPdt = 0
    dBdt = 0
    dAdt = 0

    return [dRdt, dNdt, dDdt, dPdt, dBdt, dAdt]

if __name__ == "__main__":
    # Example parameters (adjust as needed)
    C = 100.0
    Vmax = 1.0
    Vmin = -0.5
    VmaxP = 0.01
    VminP = -0.005
    VmaxB = 0.01
    VminB = -0.005
    K = 2.0
    e = 0.1
    km = 1.0
    Mmin = 0.5
    Mmax = 2.0
    pd = 0.1
    pr = 0.1
    ka = 0.5
    f = 0.05
    x = 0.1
    fNP = 0.01
    fPN = 0.01
    fNB = 0.01
    fBN = 0.01
    w = 0.05
    wB = 0.05
    di = 0.01
    dd = 0.001
    Amax = 10.0
    Dose = 5.0
    initial_dose_time = 0.0

    # Initial populations: [R, N, D, P, B, A]
    y0 = [50.0, 10.0, 0.0, 5.0, 5.0, 0.0]

    # Time span
    t = np.linspace(0, 100, 500)

    # Integrate ODEs using odeint
    sol = odeint(
        population_dynamics, y0, t,
        args=(C, Vmax, Vmin, VmaxP, VminP, VmaxB, VminB, K, e, km, Mmin, Mmax, pd, pr, ka, f, x, fNP, fPN, fNB, fBN, w, wB, di, dd, Amax, Dose, initial_dose_time)
    )

    R = sol[:, 0]
    N = sol[:, 1]
    D = sol[:, 2]
    P = sol[:, 3]
    B = sol[:, 4]
    A = sol[:, 5]

    # Print final populations
    print(f"Final populations at t={t[-1]:.1f}h:")
    print(f"Resource (R): {R[-1]:.2f}")
    print(f"Normal (N): {N[-1]:.2e}")
    print(f"Dead (D): {D[-1]:.2e}")
    print(f"Persister (P): {P[-1]:.2e}")
    print(f"Wall-deficient (B): {B[-1]:.2e}")
    print(f"Antibiotic (A): {A[-1]:.2f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.semilogy(t, R, label='Resource (R)')
    plt.semilogy(t, N, label='Normal (N)')
    plt.semilogy(t, D, label='Dead (D)')
    plt.semilogy(t, P, label='Persister (P)')
    plt.semilogy(t, B, label='Wall-deficient (B)')
    plt.semilogy(t, A, label='Antibiotic (A)')
    plt.xlabel('Time (h)')
    plt.ylabel('Population / Concentration')
    plt.title('Population Dynamics (log scale)')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(t, R, label='Resource (R)')
    plt.plot(t, N, label='Normal (N)')
    plt.plot(t, D, label='Dead (D)')
    plt.plot(t, P, label='Persister (P)')
    plt.plot(t, B, label='Wall-deficient (B)')
    plt.plot(t, A, label='Antibiotic (A)')
    plt.xlabel('Time (h)')
    plt.ylabel('Population / Concentration')
    plt.title('Population Dynamics (linear scale)')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()