# Architecture

This document describes the architecture of `tmm-core`, with particular emphasis on the separation between:

* caller-facing inputs;
* canonical problem compilation;
* backend evaluation;
* retained internal quantities;
* observable projection;
* differential decomposition;
* real-input Hermitian analysis;
* complex-input modal analysis.

The central design principle is that backends calculate a small set of algebraic primitives, while observable modules derive physical quantities from those primitives without duplicating backend-specific mathematics.

---

## 1. High-level data flow

A calculation passes through five main stages:

```text
Caller-facing input
        │
        ▼
Canonical compilation
        │
        ▼
Backend solve or retained solve
        │
        ▼
Raw observable projection
        │
        ▼
Differential decomposition
        │
        ▼
Public response
```

More concretely:

```text
CoordinateInput + Stack + Polarisation
        │
        ▼
CanonicalProblem<M, J>
        │
        ▼
Backend<J, Domain>
        │
        ├── solve  ──────► PlaneWaveSolution<Entries>
        │
        └── retain ──────► Workspace
                              │
                              ▼
                    Raw retained quantities
                              │
                              ▼
                    Observable projections
                              │
                              ▼
                 DifferentialResponseFor<...>
```

The backend is responsible for solving the planar electromagnetic problem. It is not responsible for exposing every possible physical observable.

The observable layer converts backend-independent retained quantities into amplitudes, powers, boundary states, layer-resolved quantities, integrated overlaps, and modal analysis primitives.

---

## 2. Caller-facing input and canonical compilation

Caller-facing coordinate types express physical quantities in convenient representations, including:

* spectral coordinates;
* in-plane coordinates;
* paired plane-wave coordinates;
* real-axis excitation problems;
* complex-coordinate modal problems.

Compilation transforms these representations into canonical coordinates used by every backend.

The canonical coordinate system contains the quantities required by the planar Maxwell problem, including:

```text
vacuum angular wavenumber
parallel propagation constant
```

Compilation is also responsible for:

* coordinate validation;
* canonical unit conversion;
* derivative mapping;
* jet construction;
* layer-thickness seeding;
* stack canonicalisation.

The resulting type is conceptually:

```rust
CanonicalProblem<M, J>
```

where:

* `M` is the material representation;
* `J` is the selected scalar algebra or jet representation.

The backend should not depend on the original caller-facing coordinate representation.

---

## 3. Scalar algebra and differential representation

Backend and observable calculations are generic over a scalar algebra.

A scalar algebra may represent:

* value-only sampled arrays;
* first directional derivatives;
* second directional derivatives;
* bivariate first derivatives;
* bivariate second derivatives.

Conceptually:

```text
ArrayJet0
ArrayJet1
ArrayJet2
ArrayJetBivariate1
ArrayJetBivariate2
```

The same backend and projection algebra is therefore used for values and derivatives.

Nonlinear scalar operations, including:

```text
sin
cos
exp
exprel
conjugation
Hermitian products
```

must propagate jet derivatives internally.

The observable layer initially returns raw jet-valued structures. These are subsequently decomposed into public response types by the differential assembly machinery.

```text
Raw quantity containing jets
        │
        ▼
DerivativePartsPolicy
        │
        ▼
DifferentialResponse
```

This separation prevents every physical observable from implementing its own response-layout logic.

---

## 4. Backend abstraction

A backend implements the planar electromagnetic propagation algorithm.

Current isotropic backends include:

```text
Transfer2
Scatter2
```

Both implement a shared backend interface:

```rust
Backend<J, Domain>
```

The backend produces an entry representation and may optionally retain internal calculation data.

```rust
type Entries;
type Workspace;
type Error;
```

The two primary execution modes are:

```text
solve
retain
```

### `solve`

Calculates quantities that can be projected directly from the final backend entries and exterior context.

Typical consumers include:

* reflection and transmission amplitudes;
* reflectance, transmittance, and absorptance;
* outgoing-mode determinant.

### `retain`

Retains intermediate layer data needed for internal-field and layer-resolved analysis.

Typical retained quantities include:

* per-layer propagation quantities;
* transfer prefix or scattering-cut information;
* layer boundary waves;
* layer thickness jets.

The retained workspace should contain data required by the backend calculation and immediate reconstruction. It should not become a cache for every possible future observable.

Optional analysis-specific quantities should be evaluated lazily from the canonical problem when requested.

---

## 5. Backend-independent projection traits

Backend entry representations expose physical quantities through projection traits.

Examples include:

```text
ProjectAmplitudes
ProjectPower
ProjectPlaneWaveModeDeterminant
```

These traits allow `Transfer2Entries` and `Scatter2Entries` to implement the same public semantics despite using different internal matrix representations.

Retained workspaces similarly expose reconstruction capabilities through traits such as:

```text
ReconstructLayerBoundaryWaves
RetainedIsotropicLayers
```

The evaluator depends on these capabilities rather than concrete backend types.

This enables:

* shared evaluator methods;
* cross-backend equivalence tests;
* future isotropic and anisotropic backends;
* compile-time restriction of unsupported observables.

---

## 6. Real-input and modal-input analysis

Real-input and complex-input calculations have different mathematical semantics and should remain distinct.

### Real-input analysis

Real-frequency physical observables use Hermitian products and may involve:

* complex conjugation;
* real and imaginary constitutive parts;
* signed Poynting flux;
* energy density;
* dissipation.

These methods belong on the real-input retained state.

Examples include:

```text
power
interface_power
layer_power
layer_dissipation
layer_energy
participation
confinement
```

### Complex-input modal analysis

Complex-coordinate modal calculations must preserve holomorphic dependence wherever possible.

Ordinary Hermitian products are not holomorphic in complex coordinates because they introduce complex conjugation.

Modal analysis therefore uses bilinear products between suitable left and right states.

Examples include:

```text
outgoing-mode determinant
bilinear layer overlap
mode overlap
modal norm
mode orthonormalisation
```

The modal API must not reuse real-frequency Hermitian energy types merely because their algebraic shape appears similar.

---

## 7. Boundary observables

Retained backends reconstruct directional waves at both boundaries of every finite layer.

```rust
LayerBoundaryWaves<A>
```

contains:

```text
left boundary:
    forward
    backward

right boundary:
    forward
    backward
```

The backend representation is converted into a public observable representation before further analysis.

Directional waves are transformed into canonical boundary states using:

```text
field     = forward + backward
secondary = ξ (backward - forward)
ξ         = -i Y
```

where:

```text
Y = κ / factor
```

and:

```text
factor = μ  for TE
factor = ε  for TM
```

Equivalently:

```text
secondary = factor⁻¹ ∂z field
```

The canonical state is therefore:

```text
TE:
    field     = E_y
    secondary = μ⁻¹ ∂z E_y

TM:
    field     = H_y
    secondary = ε⁻¹ ∂z H_y
```

`secondary` is not the complex conjugate of `field`.

Both components remain holomorphic under complex-coordinate evaluation.

---

## 8. Interface observables

Layer boundary states and waves are assembled into a physical interface sequence.

For a stack containing `N` finite layers, there are:

```text
N + 1 interfaces
```

Each interface contains data on both sides:

```text
Interface 0:
    left exterior
    layer 0 left boundary

Interface i:
    layer i-1 right boundary
    layer i left boundary

Interface N:
    layer N-1 right boundary
    right exterior
```

Interface-side data includes:

* directional waves;
* canonical states;
* characteristic admittance;
* interface-resolved power.

Interface assembly preserves physical left-to-right order.

The interface representation is backend-independent.

---

## 9. Power observables

Power flux uses a signed global convention:

```text
positive flux: left to right
negative flux: right to left
```

This convention is independent of incidence direction.

Interface power contains:

```text
forward-labelled flux
backward-labelled flux
net physical flux
```

Layer power is obtained from adjacent interface fluxes.

For finite layer `i`:

```text
absorbed_i
    = left boundary net flux
    - right boundary net flux
```

The formula is identical for left and right incidence.

Summing over all finite layers gives:

```text
sum(absorbed_i)
    = external absorptance
```

for a passive finite stack.

---

## 10. Analytic layer integration

Homogeneous-layer integrated observables are calculated analytically rather than through spatial sampling.

Inside one layer:

```text
forward(z)  = forward(0)  exp(+i κ z)
backward(z) = backward(0) exp(-i κ z)
```

Quadratic observables therefore reduce to four directional products:

```text
forward-forward
backward-backward
forward-backward
backward-forward
```

These are stored in:

```rust
IntegratedWaveProducts<A>
```

The exponential integral is evaluated as:

```text
∫₀ᵈ exp(αz) dz
    = d exprel(αd)
```

where:

```text
exprel(x) = (exp(x) - 1) / x
```

with its analytic continuation at zero.

`exprel` is implemented elementwise for every scalar and jet representation so mixed sampled arrays remain numerically stable.

---

## 11. Hermitian integrated wave products

Real-input integrated observables use:

```rust
integrate_hermitian_wave_products(...)
```

The left factor is complex-conjugated.

The result contains:

```text
∫ forward* forward dz
∫ backward* backward dz
∫ forward* backward dz
∫ backward* forward dz
```

The diagonal products are mathematically real.

The cross terms satisfy:

```text
backward_forward
    = conjugate(forward_backward)
```

for real layer thickness and real differentiation parameters.

Hermitian integration is used by:

* integrated state norms;
* field norms;
* dissipation;
* real-frequency energy;
* participation factors;
* confinement factors.

---

## 12. Bilinear integrated wave products

Complex modal observables use:

```rust
integrate_bilinear_wave_products(...)
```

No complex conjugation is introduced.

The function accepts distinct left and right wave sets and distinct normal propagation constants.

```text
left waves
right waves
left κ
right κ
```

This allows the caller to supply the correct primal and dual or adjoint modal solutions.

The overlap kernel itself does not decide:

* how the left mode is constructed;
* whether propagation branches are reversed;
* whether reciprocity relates left and right modes;
* which modal normalization convention is used.

Those choices belong to the modal analysis layer.

Bilinear products are used internally for:

* modal norm evaluation;
* left-right mode overlaps;
* mode orthogonalisation;
* mode orthonormalisation;
* coupled-mode projections;
* non-Hermitian perturbation theory.

They should not be exposed as a physical power or energy quantity without an explicit modal interpretation.

---

## 13. Integrated canonical-state products

Directional-wave products are projected into integrated products of the canonical state:

```rust
IntegratedStateProducts<A>
```

containing:

```text
field_field
secondary_secondary
field_secondary
secondary_field
```

For Hermitian real-input analysis:

```text
field_field
    = ∫ field* field dz

secondary_secondary
    = ∫ secondary* secondary dz

field_secondary
    = ∫ field* secondary dz

secondary_field
    = ∫ secondary* field dz
```

This layer is intentionally electromagnetic-meaning-neutral.

It only knows the canonical state definition.

TE/TM field reconstruction occurs later.

A corresponding bilinear state-product projection can operate on modal left-right wave products without introducing conjugation.

---

## 14. Integrated electromagnetic field norms

For real-input isotropic analysis, canonical state products are converted into complete vector-field norms.

```rust
IntegratedFieldNorms<R>
```

contains:

```text
electric = ∫ |E|² dz
magnetic = ∫ |H|² dz
```

The field norms are real-jet quantities.

Let:

```text
k0     = vacuum angular wavenumber
β      = parallel propagation constant
factor = μ for TE
factor = ε for TM
```

The shared Hermitian weights are:

```text
transverse weight:
    1 / |k0|²

longitudinal weight:
    |β / (k0 factor)|²
```

### TE

```text
field = E_y
```

Therefore:

```text
electric norm
    = field norm

magnetic norm
    = transverse weight × secondary norm
    + longitudinal weight × field norm
```

### TM

```text
field = H_y
```

Therefore:

```text
magnetic norm
    = field norm

electric norm
    = transverse weight × secondary norm
    + longitudinal weight × field norm
```

Cross-state products are not required for the isotropic field norms, though they remain available for more general observables.

---

## 15. Integrated layer dissipation

Integrated dissipation is represented by:

```rust
LayerDissipation<R>
```

containing:

```text
electric
magnetic
total
```

The result is normalized by the incident-wave power flux and is therefore directly comparable with:

```text
LayerPower::absorbed
```

The canonical dissipation coefficients are:

```text
electric coefficient:
    |k0|² Im(ε) / incident_flux

magnetic coefficient:
    |k0|² Im(μ) / incident_flux
```

The projected components are:

```text
electric dissipation
    = electric coefficient × integrated electric norm

magnetic dissipation
    = magnetic coefficient × integrated magnetic norm

total dissipation
    = electric + magnetic
```

The incident normalization is derived from the incident exterior admittance.

For a unit-amplitude propagating incident wave:

```text
incident flux magnitude
    = Re(Y_incident)
```

The same normalization convention is shared with the external power projection.

The principal validation identity is:

```text
LayerDissipation[i].total
    = LayerPower[i].absorbed
```

This is tested for:

* TE and TM;
* left and right incidence;
* lossless, electrically lossy, magnetically lossy, and mixed-loss layers;
* values;
* first derivatives;
* second derivatives;
* bivariate derivatives;
* transfer and scattering backends.

---

## 16. Integrated layer energy

Integrated real-frequency energy reuses:

```text
IntegratedFieldNorms
```

but applies energy-specific constitutive coefficients.

The public result should be represented by:

```rust
LayerEnergy<R>
```

with at least:

```text
electric
magnetic
total
```

Potential future extensions may include:

```text
coupling
dispersive correction
material contribution
```

Energy definitions must be explicit.

Possible definitions include:

```text
nondispersive electromagnetic energy
Brillouin dispersive energy
generalized lossy-medium energy
```

The default implementation should not silently substitute one definition for another.

For Brillouin energy, the constitutive weights involve:

```text
∂(ωε) / ∂ω
∂(ωμ) / ∂ω
```

These intrinsic constitutive derivatives are not the same as caller-requested response derivatives.

They should be evaluated lazily from the canonical stack when energy is requested.

They should not be inferred from the current response jet layout.

---

## 17. Lazy analysis data

Backend workspaces retain data needed by the backend and immediate field reconstruction.

Optional analysis-specific material data should be evaluated from:

```text
CanonicalProblem::coordinates
CanonicalProblem::stack
```

when requested.

Examples include:

* intrinsic spectral derivatives for Brillouin energy;
* higher constitutive derivatives required by energy sensitivities;
* modal normalization coefficients;
* perturbation-theory material derivatives.

This avoids increasing the cost and memory of simple calculations that only need:

* amplitudes;
* power;
* determinant;
* boundary waves.

If repeated analysis becomes expensive, an explicit prepared analysis object may cache these quantities.

```text
state.prepare_energy_analysis(...)
state.prepare_modal_analysis(...)
```

Caching should be opt-in.

---

## 18. Energy participation factors

Layer energy enables normalized participation quantities.

For each finite layer:

```text
electric participation
    = electric energy / total stack energy

magnetic participation
    = magnetic energy / total stack energy

total participation
    = layer energy / total stack energy
```

Participation factors should be represented separately from raw energy so normalization semantics remain explicit.

They may be useful for:

* resonator analysis;
* material participation;
* perturbative loss estimates;
* optimization;
* sensitivity attribution.

Zero-total-energy and non-finite normalization cases must be handled explicitly.

---

## 19. Confinement factors

A confinement factor is a normalized sum of layer energies over a selected region.

For a set of finite layers `S`:

```text
Γ_S
    = sum(U_i for i in S)
      / sum(U_i for all finite layers)
```

The API should support:

* one finite layer;
* a range of layers;
* an arbitrary selected set;
* electric-only confinement;
* magnetic-only confinement;
* total-energy confinement.

Confinement is a projection from already calculated layer energy.

It should not re-evaluate fields or materials.

---

## 20. Mode overlap matrices

A modal overlap compares two retained modal solutions.

The primitive is the bilinear integrated layer product:

```text
left mode
right mode
        │
        ▼
per-layer bilinear overlap
        │
        ▼
sum over finite layers
        │
        ▼
mode overlap
```

The result may be complex.

For a set of modes, pairwise overlaps form a matrix:

```rust
ModeOverlapMatrix<C>
```

with entries:

```text
M_ij = overlap(left_mode_i, right_mode_j)
```

The overlap definition must include:

* finite-layer contributions;
* any required exterior or boundary terms;
* constitutive derivative factors;
* the selected modal normalization convention.

The generic wave-product kernel does not impose these choices.

---

## 21. Modal normalization and orthonormalisation

Modal normalization is built from bilinear left-right products.

For one mode pair:

```text
N_i = overlap(left_mode_i, right_mode_i)
```

A right mode may be normalized by applying a scalar factor:

```text
right_mode_i
    → right_mode_i / N_i
```

or a square-root convention, depending on the selected left-right normalization.

For several non-orthogonal modes, construct the overlap matrix:

```text
M_ij = overlap(left_i, right_j)
```

Orthonormalisation should use this matrix internally.

Possible methods include:

* biorthogonal Gram–Schmidt;
* overlap-matrix factorization;
* singular-value decomposition;
* symmetric inverse-square-root normalization.

The implementation should not assume ordinary Hermitian orthogonality.

A robust default for non-Hermitian mode sets is based on an overlap matrix factorization.

Conceptually:

```text
Lᴮ R = M
```

Construct transformed mode bases such that:

```text
L'ᴮ R' = I
```

where `ᴮ` denotes the selected bilinear left-right pairing, not Hermitian conjugation.

The public API should expose normalized modes and diagnostics, not raw internal bilinear products unless requested explicitly.

Diagnostics should include:

* original overlap matrix;
* condition number or singular values;
* rank deficiency;
* normalization residual;
* modes rejected as linearly dependent.

---

## 22. Differential decomposition of integrated observables

Every integrated observable should support the same differential response layouts as the rest of the evaluator.

Examples include:

```text
LayerEnergy
LayerDissipation
LayerParticipation
LayerConfinement
ModeOverlap
ModeNorm
```

Raw jet-valued containers implement:

```text
IntoValue
IntoFirst
IntoSecond
IntoBivariateFirst
IntoBivariateSecond
```

Generic sequence containers such as:

```rust
Layers<T>
```

transpose the derivative structure across all finite layers.

The evaluator then assembles:

```rust
DifferentialResponseFor<J, Quantity<J>>
```

using the selected mapping and policy.

Observable modules should not manually construct public derivative response layouts.

---

## 23. Public evaluator API

The evaluator exposes low-cost quantities on solved states and retained quantities on retained states.

### Solved state

Typical methods:

```text
amplitudes
power
determinant
```

### Retained real-input state

Typical methods:

```text
boundary_waves
boundary_states
interface_power
layer_power
layer_dissipation
layer_energy
participation
confinement
```

### Retained modal state

Typical methods:

```text
bilinear_layer_overlap
mode_overlap
mode_norm
normalise_mode
orthonormalise_modes
```

Methods should be present only where their mathematical semantics are valid.

Hermitian physical observables should remain unavailable on complex modal states.

---

## 24. Prepared evaluation

Repeated atomic evaluation, such as adaptive contour integration or optimization, should avoid recompiling static problem data.

The intended prepared pipeline is:

```text
prepare stack and derivative mapping once
        │
        ▼
compile only new canonical coordinates
        │
        ▼
evaluate backend
        │
        ▼
project requested scalar quantity
```

A prepared modal problem may expose:

```text
determinant_at
amplitudes_at
mode_data_at
retain_at
```

A prepared real-input optimization problem may expose:

```text
power_at
layer_dissipation_at
layer_energy_at
```

Prepared evaluation and observable projection are complementary:

* preparation removes repeated structural compilation;
* projection derives only the requested physical quantity.

---

## 25. Backend equivalence strategy

Backend equivalence is established at the lowest shared semantic layer practical.

Current equivalence tests compare:

* amplitudes;
* power;
* mode determinant;
* retained layer boundary waves;
* derivative branches.

Once two backends produce equivalent raw boundary waves and retained isotropic quantities, higher observable projections need not be duplicated exhaustively for both backends.

The recommended strategy is:

```text
full observable suite:
    primary stable backend

smoke tests:
    secondary backend

raw retained equivalence:
    all backends
```

For optically thick or strongly absorbing stacks, the scattering backend is the authoritative stable implementation.

---

## 26. Testing hierarchy

Tests should be organized by abstraction level.

### Algebra tests

Validate:

* jet arithmetic;
* nonlinear unary functions;
* `exprel`;
* real/imaginary projections;
* Hermitian products.

### Analytic integration tests

Validate:

* closed-form exponential integration;
* numerical quadrature agreement;
* zero-thickness limits;
* zero-exponent limits;
* first and second derivatives;
* bivariate Hessian terms;
* Hermitian conjugation symmetry;
* bilinear products.

### Pure projection tests

Validate:

* component routing;
* TE/TM dispatch;
* sequence order;
* metadata preservation;
* derivative decomposition.

### Evaluator-level physical tests

Validate:

* interface continuity;
* flux conservation;
* layer power telescoping;
* dissipation equals flux loss;
* energy identities;
* participation normalization;
* confinement normalization;
* modal overlap normalization.

### Backend equivalence tests

Validate shared raw semantics between backend implementations.

---

## 27. Module layout

The intended observable layout is:

```text
observable/
├── boundary/
│   ├── mod.rs
│   ├── waves.rs
│   ├── state.rs
│   └── project.rs
├── interface/
│   ├── mod.rs
│   ├── data.rs
│   ├── power.rs
│   └── project.rs
├── layer/
│   ├── mod.rs
│   ├── data.rs
│   ├── power.rs
│   ├── overlap.rs
│   ├── state_overlap.rs
│   ├── field_norm.rs
│   ├── dissipation.rs
│   ├── energy.rs
│   ├── participation.rs
│   ├── confinement.rs
│   └── project.rs
└── modal/
    ├── mod.rs
    ├── overlap.rs
    ├── norm.rs
    ├── matrix.rs
    └── orthonormalise.rs
```

Derivative decomposition implementations mirror the observable layout:

```text
derivative_parts/
└── quantity/
    ├── boundary/
    ├── interface/
    ├── layer/
    └── modal/
```

Evaluator tests similarly follow the public analysis surface:

```text
evaluate/tests/
├── solve/
├── retain/
│   ├── boundary_waves.rs
│   ├── interface_power.rs
│   ├── layer_power.rs
│   ├── layer_dissipation.rs
│   ├── layer_energy.rs
│   ├── participation.rs
│   └── confinement.rs
└── modal/
    ├── overlap.rs
    ├── norm.rs
    └── orthonormalisation.rs
```

---

## 28. Planned implementation sequence

The next integrated observable work should proceed in this order.

### 1. Integrated layer energy

Build:

```text
LayerEnergy
EnergyDefinition
energy-specific constitutive analysis data
real-input evaluator projection
```

Validate:

```text
lossless nondispersive limits
positive passive energy
layer sum
derivative propagation
```

### 2. Electric and magnetic participation factors

Build:

```text
LayerParticipation
stack totals
normalization error handling
```

Validate:

```text
sum(total participation) = 1
sum(electric participation) = 1
sum(magnetic participation) = 1
```

where the corresponding denominator is nonzero.

### 3. Confinement factors

Build:

```text
LayerSelection
ConfinementFactor
electric, magnetic, total variants
```

Validate:

```text
all layers → 1
empty selection → 0
disjoint regions add
```

### 4. Modal overlap matrices

Build on complex retained states and:

```text
integrate_bilinear_wave_products
```

Define:

```text
LayerBilinearOverlap
ModeOverlap
ModeOverlapMatrix
```

Keep the pairing convention explicit.

### 5. Modal normalization and orthonormalisation

Use the overlap matrix internally.

Provide:

```text
normalize one mode
biorthogonalize several modes
orthonormalisation diagnostics
```

Do not implement modal orthonormalisation through Hermitian products.

---

## 29. Design invariants

The architecture should preserve the following invariants.

### Canonical compilation

```text
backends receive only canonical problems
```

### Backend independence

```text
physical observable semantics do not depend on backend representation
```

### Real versus modal separation

```text
Hermitian products belong to real-input physical analysis
bilinear products belong to complex modal analysis
```

### Analytic integration

```text
integrated homogeneous-layer quantities do not require spatial sampling
```

### Lazy optional analysis

```text
energy-specific and modal-analysis material data are evaluated only when requested
```

### Differential consistency

```text
values and derivatives use the same physical projection algebra
```

### Normalization consistency

```text
layer dissipation, layer power, and external absorptance use the same incident-flux convention
```

### Compiler-visible semantics

```text
types and capability traits prevent unsupported observable/backend/domain combinations
```

These invariants should guide future additions, including anisotropic backends, spatial sampling, prepared evaluation, adjoint sensitivities, and modal optimization.
