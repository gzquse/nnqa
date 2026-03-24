# Cloud Job Resource Estimation

## Summary

**Total Jobs:** 60 (6 degrees × 10 trials)

**Total Iterations:** 900 (15 iterations per job × 60 jobs)
- Each iteration = 1 parameterized circuit

**Shots per Iteration:** 4,096

**Total Shots:** 3,686,400

---

## Table 1: Direct Approach (1 Qubit) - Current Implementation

**Method:** Polynomial computed classically, result encoded into single qubit.

| Degree | Qubits | 1Q Gates/Iter | 2Q Gates/Iter | Iterations | Shots/Iter |
|--------|--------|---------------|--------------|------------|------------|
| 1      | 1      | 1             | 0            | 150        | 4,096      |
| 2      | 1      | 1             | 0            | 150        | 4,096      |
| 3      | 1      | 1             | 0            | 150        | 4,096      |
| 4      | 1      | 1             | 0            | 150        | 4,096      |
| 5      | 1      | 1             | 0            | 150        | 4,096      |
| 6      | 1      | 1             | 0            | 150        | 4,096      |
| **TOTAL** | **1** | **1** | **0** | **900** | **4,096** |

**Key:** All resources are **constant** regardless of polynomial degree!

---

## Table 2: Native Approach (Multi-Qubit) - Alternative

**Method:** Uses quantum multiplication to compute polynomial powers natively.

| Degree | Qubits | 1Q Gates/Iter | 2Q Gates/Iter | Iterations | Shots/Iter |
|--------|--------|---------------|--------------|------------|------------|
| 1      | 2      | 1             | 0            | 150        | 4,096      |
| 2      | 3      | 3             | 1            | 150        | 4,096      |
| 3      | 4      | 5             | 2            | 150        | 4,096      |
| 4      | 5      | 7             | 3            | 150        | 4,096      |
| 5      | 6      | 9             | 4            | 150        | 4,096      |
| 6      | 7      | 11            | 5            | 150        | 4,096      |
| **RANGE** | **2-7** | **1-11** | **0-5** | **900** | **4,096** |

**Key:** Resources **scale linearly** with polynomial degree.

---

## Per Iteration Details

### Direct Approach (1 Qubit)
- **Qubits per iteration:** 1 (constant)
- **1-qubit gates per iteration:** 1 RY (constant)
- **2-qubit gates per iteration:** 0 (constant)
- **Shots per iteration:** 4,096 (constant)

### Native Approach (Multi-Qubit)
- **Qubits per iteration:** 2-7 (scales with degree)
- **1-qubit gates per iteration:** 1-11 (scales with degree)
  - RY gates: 1-6
  - RZ gates: 0-5
- **2-qubit gates per iteration:** 0-5 CNOT (scales with degree)
- **Shots per iteration:** 4,096 (constant)

---

## Per Job Resources (15 Iterations)

### Direct Approach
- **Iterations:** 15
- **Total shots:** 61,440
- **Total gates:** 15 (all 1-qubit RY)
- **Qubits:** 1 (same qubit reused)

### Native Approach
- **Iterations:** 15
- **Total shots:** 61,440
- **Total gates:** 15-165 (depending on degree)
- **Qubits:** 2-7 (depending on degree)

---

## Circuit Structure

### Direct Approach (All Degrees)
```
1. Classical: y = F(x) = a0 + a1*x + a2*x^2 + ...
2. Quantum: Ry(arccos(y))|0⟩ on qubit 0
3. Measure: <Z> = y
```
- **Qubits:** 1
- **Gates:** 1 RY
- **Depth:** 1 gate

### Native Approach (Degree d)
```
1. Encode x on qubit 0: Ry(arccos(x))|0⟩
2. Compute x^2: quantum multiplication (qubits 0,1)
3. Compute x^3: quantum multiplication (qubits 1,2)
4. ... (continue for higher powers)
5. Measure highest power qubit
```
- **Qubits:** d+1
- **Gates:** 2d-1 one-qubit, d-1 two-qubit
- **Depth:** ~3d gates

---

## Key Observations

### Direct Approach
1. **Constant resources:** 1 qubit, 1 gate regardless of degree
2. **No two-qubit gates:** Simpler execution, less error
3. **Resource efficient:** Minimal hardware requirements
4. **Same error characteristics:** Shot noise only

### Native Approach
1. **Linear scaling:** Resources grow with degree
2. **Quantum arithmetic:** Demonstrates native quantum computation
3. **Higher gate count:** More complex execution
4. **Same error characteristics:** Shot noise only

---

## For Paper

**Current implementation uses Direct Approach:**
- Qubits per iteration: 1 (constant)
- Gates per iteration: 1 (1 RY, constant)
- Shots per iteration: 4,096
- Iterations per job: 15
- Total iterations: 900

**Statement:**
"Our experiments use a direct encoding approach where polynomials are computed classically and results are encoded into a single-qubit quantum state. This approach uses constant resources (1 qubit, 1 gate per iteration) regardless of polynomial degree, demonstrating that quantum encoding/decoding fidelity is independent of polynomial complexity."
