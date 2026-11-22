# Project Specification: Safe-by-Design Nanotoxicology with DiCE

## 1. Project Overview & Objective

Nanomaterials hold immense promise as next-generation antimicrobial agents against multi-drug resistant (MDR) bacterial pathogens. However, clinical and environmental translation is frequently bottlenecked by toxicological liabilities against mammalian host cells (e.g., membrane rupture, cytotoxic reactive oxygen species generation, loss of cellular viability).

The primary objective of this project is to implement a **Safe-by-Design (SbD)** framework that:
- **Reduces mammalian cytotoxicity** (elevating mammalian cell viability above biocompatibility thresholds).
- **Preserves antibacterial potency** against targeted MDR pathogens (maintaining minimum inhibitory concentration, or MIC, within effective therapeutic windows).
- Employs **Constrained Diverse Counterfactual Explanations (DiCE)** to discover minimal, physically realizable, and chemically actionable modifications that navigate the trade-off between antibacterial efficacy and biocompatibility.

---

## 2. Core Descriptors

The physicochemical profile of each nanomaterial is captured using five foundational design features:

1. **Core Material**: The elemental or compositional core of the nanoparticle (e.g., $Ag$, $ZnO$, $CuO$, $Au$, $Fe_3O_4$).
2. **Particle Size ($nm$)**: The primary/hydrodynamic core diameter.
3. **Zeta Potential ($mV$)**: The surface charge density influencing electrostatic interactions with anionic bacterial membranes and mammalian cell surfaces.
4. **Surface Coating / Capping Agent**: The surface functionalization or ligand layer (e.g., citrate, PEG, PVP, branched PEI, CTAB).
5. **Shape / Morphology**: Nanoparticle geometry (e.g., spherical, rod, wire, star, sheet), which governs contact surface area and cellular uptake kinetics.

---

## 3. Dual Optimization Targets

Optimization is formulated around dual endpoints:

| Endpoint | Target Metric | Objective | Design Criterion |
| :--- | :--- | :--- | :--- |
| **Mammalian Safety** | Cell Viability ($\%$) | Maximize ($\ge \text{threshold}$) | Biocompatible / Low cytotoxicity (e.g., viability $\ge 80\%$) |
| **Antimicrobial Efficacy**| Bacterial MIC ($\mu g/mL$) | Minimize ($\le \text{threshold}$) | Potent pathogen suppression (e.g., MIC $\le 16\,\mu g/mL$) |

Both regression (predicting continuous viability and MIC values) and classification (categorizing into biocompatible vs. toxic, sensitive vs. resistant) formulations will be supported.

---

## 4. Baseline Machine Learning Models

To establish robust property-activity relationship baselines prior to counterfactual generation, three tree ensemble architectures are implemented and benchmarked:

1. **Random Forest (RF)**: Robust bagging ensemble providing stable non-linear baselines and resistance to overfitting on small-to-moderate tabular cohorts.
2. **XGBoost**: Extreme Gradient Boosting optimized for structured tabular features, capturing non-linear interactions between core composition and surface charge.
3. **CatBoost**: Categorical Boosting native handling of categorical features (core material, surface coating, shape) with minimal target leakage.

Models are validated using stratified grouped cross-validation with standardized evaluation metrics ($ROC\text{-}AUC$, $F_1$, Balanced Accuracy, $R^2$, and $RMSE$).

---

## 5. Counterfactual Search & Domain Constraints (DiCE)

Counterfactual generation via DiCE explores the perturbation space to answer: *"What is the minimal, synthetically viable change to this toxic nanoparticle candidate that renders it safe while maintaining antibacterial activity?"*

### Constraint Categorization

1. **Fixed (Immutable) Features**:
   - `core_material`: Cannot be altered post-synthesis without redefining the chemical identity and fundamental electronic properties of the nanomaterial.
   - `bacterial_strain`: The pathogen target is an environmental condition and problem specification, not an engineerable nanomaterial property.

2. **Tunable (Mutable) Levers**:
   - `particle_size_nm`: Controlled via synthesis reaction time, precursor concentration, or temperature (bounded within physical synthesizability ranges).
   - `zeta_potential_mv`: Adjusted through pH modification, surface grafting, or varying ligand density (bounded within colloidal stability limits).
   - `surface_coating`: Modifiable via ligand exchange, polymer encapsulation, or functional group substitution.

3. **Configurable Levers**:
   - `shape`: Flagged as configurable. While certain synthesis routes strictly yield specific geometries (making shape immutable for that route), other wet-chemical protocols allow shape-switching (e.g., spheres vs. rods), allowing users to toggle shape between mutable and fixed based on laboratory synthesis capabilities.

---

## 6. Evaluation Framework

Generated counterfactuals will be evaluated across four dimensions:
- **Proximity ($L_1 / L_2$ distance)**: Minimizing the magnitude of change required from the original candidate.
- **Sparsity**: Minimizing the count of modified features (preferring single-lever changes, e.g., swapping coating only).
- **Diversity**: Providing multiple distinct, orthogonal redesign candidates per input.
- **Feasibility & Synthesizability**: Enforcing strictly valid physical bounds and valid categorical domain values.
