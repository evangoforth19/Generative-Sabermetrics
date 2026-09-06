# Bayesian Research Project Context

## 1. Project Overview
This project aims to build a **physics-informed Bayesian model of baseball contact**, specifically to decompose and explain **exit velocity (EV)** through underlying physical mechanisms and latent variables.

The core goal is to:
- Reconstruct the **data-generating process of bat-ball contact**
- Separate **signal (true physics)** from **noise (measurement + model error)**
- Build a **generative, probabilistic simulation of baseball contact**

Motivation:
- Existing models are either **misspecified (rigid-body simplifications)** or **uninterpretable (black-box ML)**
- Statcast contains enough structure to recover latent physics, but only under **careful probabilistic modeling**
- The long-term objective is a **scalable, interpretable “world model” of baseball contact**

### Project Evolution
The work has progressed in structured phases:

1. **EV Distribution Exploration**
   - Conducted in the `EV Distribution Search` folder
   - Goal: understand the **shape, skew, and variance structure** of exit velocity
   - Insight: EV distributions exhibit **non-Gaussian behavior**, motivating heavy-tailed likelihoods and latent structure

2. **MCMC1**
   - First attempt at Bayesian reconstruction of contact physics
   - Established:
     - Feasibility of inferring `x` and `e`
     - Initial structure of likelihood and priors
   - Limitations:
     - Weak identifiability
     - Poor handling of noise and geometry

3. **MCMC2 Iterative Framework**
   - Sequential refinement of the model (e.g., `MCMC2_1 → ... → MCMC2_6`)
   - Each iteration introduces:
     - Improved filtering
     - Better priors
     - More stable geometry
   - Core philosophy:
     - **Fix the data first, then fix the model**

4. **Current Direction**
   - Focus on:
     - **Filtering events with clean normal direction representations**
     - Stabilizing inference of `x` and `e`
   - Then:
     - Systematically extend the physics to include **tangential effects (spin, friction, obliquity)**

---

## 2. Core Modeling Framework

### Bayesian Approach
- Physics-informed Bayesian inference
- Combines:
  - Deterministic collision equations
  - Latent variable inference
  - Hierarchical priors
  - Measurement error modeling

### Key Model: MCMC2
- Current working model family (latest: `MCMC2_6`)
- Estimates:
  - Event-level:
    - `x` (contact location)
    - `e` (restitution)
  - Hitter-level:
    - `x_cm`, `r_g`

### Inference
- Implemented in PyMC
- Sampling via MCMC (NUTS)
- Likelihood:
  - Based on **normal component of EV**
  - Typically **Student-t** (heavy-tailed robustness)

### Structure
- Hierarchical:
  - Events share underlying physics
  - Hitter-level parameters partially pooled
- Sequential refinement:
  - Each MCMC2 version builds directly on the previous

---

## 3. Key Variables and Concepts

### Core Physical Quantities
- `x`: Contact location along the bat
- `e`: Coefficient of restitution
- `n`: Contact normal vector
- `v_ball_in`, `v_bat`, `v_out`: velocity vectors

### Normal Collision Model
\[
v_{f,n} = v_{ball,n} + \frac{1 + e}{1 + R_0(x)} (v_{bat,n} - v_{ball,n})
\]

\[
R_0(x) = \frac{m}{M} \left(1 + \frac{(x - x_{cm})^2}{r_g^2} \right)
\]

### Latent Variables
- `x`, `e`
- Future:
  - True normal vector
  - Tangential interaction regime

### Derived Quantities
- `v_f_obs_n = v_out · n`
- `v_ball_n`, `v_bat_n`
- Obliquity angle
- Normal EV fraction

---

## 4. Data and Feature Construction

### Data
- Statcast pitch-level data
- Filtered BIP events:
  - `type == 'X'`
  - `launch_speed ≥ 40`
  - `bat_speed ≥ 40`

### Feature Engineering

#### Velocity Construction
- Ball:
  \[
  v_{in} = v_0 + a \cdot t_{contact}
  \]
- Bat:
  - From bat speed + attack angles
- Outgoing:
  - From launch + spray angles

#### Normal Vector
\[
n_0 = \frac{v_{rel,out} - v_{rel,in}}{\|v_{rel,out} - v_{rel,in}\|}
\]

#### Time to Contact
- Quadratic solve in vertical dimension
- Linear fallback if unstable

#### Transformations
- Coordinate shifts (spray angle)
- Unit conversions (mph → fps, deg → rad)

---

## 5. Model Structure and Assumptions

### Assumptions
- Contact approximated by rigid-body physics (baseline)
- EV primarily driven by normal component
- Normal vector can be approximated from observed data
- Noise is heavy-tailed and non-trivial

### Limitations
- Normal vector proxy breaks in edge cases
- Tangential physics ignored
- Restitution assumed overly simple
- Non-linear effects missing

### Uncertainty Sources
- Measurement noise
- Latent variable ambiguity (`x`, `e`)
- Geometric instability
- Model misspecification

---

## 6. Current Bottlenecks and Open Questions

### Bottlenecks
- MCMC runtime
- Identifiability between `x` and `e`
- Sensitivity to normal vector estimation
- Noise amplification through transformations

### Model Failures
- Underestimates high EV
- Residual structure unexplained
- Instability in edge geometries

### Open Questions
- Is the normal vector fundamentally wrong?
- Should `e` be context-dependent?
- Are missing interaction terms critical?
- Can latent geometry be directly modeled?

---

## 7. Desired Extensions

### Near-Term
- Aggressive filtering for **clean normal-direction events**
- Stabilize inference of `x` and `e`

### Next Phase
- Extend equation to include:
  - Tangential velocity components
  - Spin / friction effects
  - Obliquity-dependent behavior

### Modeling
- Context-dependent priors
- Mixture models for contact regimes
- Gaussian Process prior learning

### Long-Term
- Full generative contact simulation
- Latent normal vector inference
- Model discrepancy correction:
  \[
  \gamma(z) = f(x, n, v_{ball}, v_{bat})
  \]

---

## 8. Code Structure

### Workflow
- `EV Distribution Search/`
  - Exploratory analysis of EV distributions

- `MCMC1/`
  - Initial Bayesian model

- `MCMC2/`
  - Iterative refinement
  - Versions (`MCMC2_1` → `MCMC2_6`) are **sequential and cumulative**

### Pipeline
1. Raw data → `bip_df`
2. Feature engineering → `bip_sub`
3. Clean dataset → `bip_model`
4. PyMC model definition
5. MCMC sampling
6. Diagnostics (ArviZ)

---

## 9. OpenBiomechanics Integration Goal

### Objective
Use the **openbiomechanics repository** to:
- Improve understanding of:
  - Contact geometry
  - Bat kinematics
- Validate and enhance inference of:
  - `x` (contact location)
  - `e` (restitution)

### Approach
- Apply **same MCMC architecture (MCMC2_6)** using:
  - Nathan-style equations
  - Biomechanics-derived inputs

### Key Goals
1. Estimate:
   - Covariance structure between:
     - `x`, `e`
     - Input variables (bat speed, angles, etc.)
2. Quantify:
   - Posterior uncertainty in `x` and `e`
   - Distributional shape under cleaner measurement conditions
3. Compare:
   - Statcast-derived inference vs biomechanics-derived inference

### Data Usage
- Focus on **hitter datasets**
- Use `.readme` files to:
  - Identify available markers
  - Understand bat tracking structure
  - Extract usable kinematic variables

### Expected Benefit
- Reduce reliance on noisy proxy constructions
- Provide **ground-truth-informed priors**
- Improve identifiability and physical realism

---

## 10. How to Compare This Project to Other Repositories

### Relevant If They Provide

#### Physics
- Bat-ball collision modeling
- Contact mechanics

#### Probabilistic Methods
- Bayesian inference
- Hierarchical models
- Latent variable estimation

#### Geometry / Kinematics
- 3D reconstruction
- Motion capture processing
- Vector-based modeling

#### Data
- Biomechanics datasets (C3D)
- Bat tracking

#### Simulation
- Generative systems
- Monte Carlo engines

### Definition of Relevance
A repo is useful if it:
- Improves identifiability of `x` or `e`
- Reduces model misspecification
- Enhances physical interpretability
- Enables better priors or constraints
- Supports generative simulation

### Not Relevant
- Pure black-box ML
- Surface-level analytics
- Visualization-only tools

---

This project is a **Bayesian reconstruction of a physical system under uncertainty**, not a prediction pipeline. Any external resource should be evaluated based on whether it helps recover the true underlying mechanics of contact.