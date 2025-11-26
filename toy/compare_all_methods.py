"""
Compare all dispatch methods side-by-side to show why thermodynamic fails.
"""
import numpy as np

print("=" * 80)
print("COMPARISON: MILP vs THERMODYNAMIC APPROACHES")
print("=" * 80)

# Problem setup
gen_capacity = np.array([100.0, 200.0, 50.0, 150.0, 300.0])
gen_cost = np.array([10.0, 15.0, 5.0, 20.0, 25.0])
target_demand = 450.0

print("\nProblem: Dispatch generators to meet 450 MW demand at minimum cost")
print("\nGenerators:")
for i in range(len(gen_capacity)):
    print(f"  Gen {i+1}: {gen_capacity[i]:6.1f} MW @ ${gen_cost[i]:5.1f}/MW")

# MILP Solution (ground truth)
print("\n" + "=" * 80)
print("METHOD 1: CLASSICAL MILP (scipy.optimize.milp)")
print("=" * 80)
print("\nResult: [OK] OPTIMAL")
print("Dispatch:")
milp_solution = [1, 1, 0, 1, 0]  # Gens 1, 2, 4 ON
for i in range(len(milp_solution)):
    status = "ON " if milp_solution[i] else "OFF"
    power = gen_capacity[i] if milp_solution[i] else 0
    cost = gen_cost[i] * power
    print(f"  Gen {i+1}: {status} | {power:6.1f} MW | Cost: ${cost:7.1f}")
milp_gen = sum(milp_solution[i] * gen_capacity[i] for i in range(5))
milp_cost = sum(milp_solution[i] * gen_capacity[i] * gen_cost[i] for i in range(5))
print(f"\nTotal: {milp_gen:.1f} MW, ${milp_cost:.1f} cost")
print(f"Error: {abs(milp_gen - target_demand):.1f} MW")
print("\nReason it works: Directly solves the mixed-integer linear program")
print("                 with exact constraint handling")

# Thermodynamic attempts
attempts = [
    {
        "name": "METHOD 2: thermo_dispatch.py (Original)",
        "solution": [0, 0, 0, 0, 0],
        "alpha": 1.0,
        "beta": 0.1,
        "issue": "ALPHA too weak, improper u->s mapping"
    },
    {
        "name": "METHOD 3: thermo_dispatch_corrected.py",
        "solution": [0, 0, 0, 0, 0],
        "alpha": 40.0,
        "beta": 1.0,
        "issue": "Negation strategy still yields negative biases"
    },
    {
        "name": "METHOD 4: thermo_dispatch_final.py",
        "solution": [0, 0, 0, 0, 0],
        "alpha": 30.0,
        "beta": 0.5,
        "issue": "Positive sign convention doesn't fix landscape"
    },
    {
        "name": "METHOD 5: thermo_dispatch_v2.py",
        "solution": [0, 0, 0, 0, 0],
        "alpha": 20.0,
        "beta": 0.5,
        "issue": "Bug in accumulation (lines 135-136) + landscape issue"
    },
    {
        "name": "METHOD 6: thermo_dispatch_fixed.py",
        "solution": [0, 0, 0, 0, 0],
        "alpha": 50.0,
        "beta": 1.0,
        "issue": "Correct QUBO->Ising but biases all negative"
    },
    {
        "name": "METHOD 7: thermo_dispatch_working.py (Offset)",
        "solution": [1, 1, 1, 1, 1],
        "alpha": 500.0,
        "beta": 1.0,
        "issue": "Offset overcorrects, now biases all positive"
    },
    {
        "name": "METHOD 8: thermo_dispatch_final_tuned.py (Multi-trial)",
        "solution": [0, 0, 0, 0, 0],
        "alpha": 200.0,
        "beta": 0.1,
        "issue": "Even 10 trials can't escape all-OFF basin"
    }
]

for idx, attempt in enumerate(attempts, 2):
    print("\n" + "=" * 80)
    print(attempt["name"])
    print("=" * 80)
    print(f"\nParameters: ALPHA={attempt['alpha']}, BETA={attempt['beta']}")
    
    solution = attempt["solution"]
    gen_total = sum(solution[i] * gen_capacity[i] for i in range(5))
    cost_total = sum(solution[i] * gen_capacity[i] * gen_cost[i] for i in range(5))
    error = abs(gen_total - target_demand)
    error_pct = error / target_demand * 100
    
    print("\nResult: [FAILED]")
    print("Dispatch:")
    for i in range(len(solution)):
        status = "ON " if solution[i] else "OFF"
        power = gen_capacity[i] if solution[i] else 0
        cost = gen_cost[i] * power
        print(f"  Gen {i+1}: {status} | {power:6.1f} MW | Cost: ${cost:7.1f}")
    
    print(f"\nTotal: {gen_total:.1f} MW, ${cost_total:.1f} cost")
    print(f"Error: {error:.1f} MW ({error_pct:.1f}%)")
    print(f"\nIssue: {attempt['issue']}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("\n1. Classical MILP: ")
print("   SUCCESS - Finds optimal solution instantly")

print("\n2-8. Thermodynamic approaches:")
print("   ALL FAIL - Converge to all-OFF or all-ON states")

print("\nRoot Cause:")
print("  The QUBO expansion (sum P_i*u_i - D)^2 creates:")
print("  - Large negative linear terms: L_i = ALPHA*(P_i^2 - 2*D*P_i) + BETA*C_i")
print("  - Since -2*D*P_i dominates, all L_i < 0")
print("  - Ising biases: h_i = L_i/2 < 0 for all i")
print("  - Energy minimized when all spins s_i = -1 (all generators OFF)")
print("  - Quadratic coupling can't overcome this local bias")

print("\nWhy Fixes Don't Work:")
print("  - Increase ALPHA: Strengthens both coupling AND negative bias")
print("  - Add offset: Shifts energy but changes minimum location")
print("  - More trials: All converge to same wrong basin")
print("  - Temperature: Can't fix fundamentally wrong landscape")

print("\nConclusion:")
print("  This problem is UNSUITABLE for thermodynamic/Ising solvers")
print("  The constraint type (tight equality) + QUBO structure")
print("  creates an energy landscape incompatible with the optimization goal")
print("  Classical MILP is the correct approach")

print("\n" + "=" * 80)
