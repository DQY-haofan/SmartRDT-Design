# Ontology Ablation Study Report (Improved)

**Generated**: 2025-12-17 23:52:42

## 1. Summary

This improved ablation study uses **soft ablation** (noise injection) in addition to 
hard ablation (feature disabling) to quantify the marginal value of each ontology component.

## 2. Results Table

| Mode | Category | Feasible Rate | Max Recall | Pareto Size |
|------|----------|--------------|------------|-------------|
| Full Ontology (Baseline)  | baseline | 11.7% | 0.960 | 34 |
| Property Noise 10%        | soft     | 9.2% | 0.960 | 66 |
| Property Noise 30%        | soft     | 8.1% | 0.960 | 57 |
| Property Noise 50%        | soft     | 7.6% | 0.960 | 47 |
| No Compatibility Rules    | hard     | 12.7% | 0.960 | 31 |
| Type Defaults Only        | hard     | 15.7% | 0.960 | 40 |
| Degraded 50%              | combined | 3.7% | 0.960 | 38 |

## 3. Key Findings

### 3.1 Property Query Sensitivity

Property queries retrieve component-specific attributes from the ontology.
Adding noise to these queries simulates real-world uncertainty in component specifications.

| Noise Level | Feasible Rate | Δ vs Baseline |
|-------------|--------------|---------------|
| 0% | 11.7% | +0.0% |
| 10% | 9.2% | -21.9% |
| 30% | 8.1% | -31.2% |
| 50% | 7.6% | -34.9% |

## 4. Paper-Ready Conclusions

### For Methods Section:

> The ontology ablation study evaluates the contribution of each knowledge representation 
> component. We employ both hard ablation (complete feature removal) and soft ablation 
> (noise injection) to quantify the sensitivity of optimization performance to ontology accuracy.

### For Results Section:

> Property queries from the ontology are **critical** for optimization success. 
> Adding 30% noise to property values reduces feasible solution discovery by approximately X%.
> Disabling compatibility rules has a moderate impact (Y% change in feasible rate),
> while type constraints show minimal direct impact on solution quality.

## 5. Ablation Mode Details

### Full Ontology (Baseline)
- **Category**: baseline
- Property Noise: 0%
- Type Defaults: No
- Compatibility: Enabled

### Property Noise 10%
- **Category**: soft
- Property Noise: 10%
- Type Defaults: No
- Compatibility: Enabled

### Property Noise 30%
- **Category**: soft
- Property Noise: 30%
- Type Defaults: No
- Compatibility: Enabled

### Property Noise 50%
- **Category**: soft
- Property Noise: 50%
- Type Defaults: No
- Compatibility: Enabled

### No Compatibility Rules
- **Category**: hard
- Property Noise: 0%
- Type Defaults: No
- Compatibility: Disabled

### Type Defaults Only
- **Category**: hard
- Property Noise: 0%
- Type Defaults: Yes
- Compatibility: Enabled

### Degraded 50%
- **Category**: combined
- Property Noise: 50%
- Type Defaults: Yes
- Compatibility: Disabled

