# Phase 9: Startup Digital Twin
## Step 17: Build Startup Simulator

**Objective:** Create a digital twin simulator that responds to user actions, updating core startup metrics and predicting failure risk using the `failure_prediction_model.joblib` artifact.

### 1. File Creation & Setup
- [ ] Create `simulation/startup_twin.py`.
- [ ] Import dependencies (e.g., `pandas`, `numpy`, `joblib`).
- [ ] Load the persisted failure model (`models/failure_prediction_model.joblib`).
- [ ] Initialize the Digital Twin state structure (e.g., `team_size`, `launched_at`, `cohort_year`, `description`/`one_liner`, `revenue`, `funding`, etc.).

### 2. Implement Feature Featurizer (Model Contract)
- [ ] Implement a function (like `_featurize` from the model) to safely extract the required SAFE features from the startup's current state:
  - `team_size`
  - `months_since_launch` (derived from `launched_at` or `cohort_year`)
  - `text` (concatenated `one_liner` and `description`)
- [ ] Ensure the simulator maps Twin states into the format expected by the loaded model.

### 3. Implement User Actions
Create functions to simulate the startup's lifecycle events and business decisions. Each action updates specific state variables:
- [ ] **Hire Employees:** 
  - Increases `team_size` and recalculates `log_team_size`.
  - Decreases cash runway / impacts Funding.
- [ ] **Raise Funding:** 
  - Increases total `funding` and extends cash runway.
  - Generates Funding Impact metrics.
- [ ] **Increase Marketing Budget:** 
  - Depletes current funding.
  - Generates Growth Impact and Revenue Impact over time.
- [ ] **Launch Product:** 
  - Sets `launched_at` timestamp.
  - Activates the `months_since_launch` feature used heavily in risk prediction.
- [ ] **Expand Market / Pivot:** 
  - Updates the `description` or `one_liner` text.
  - Can be integrated later with the Pivot Recommendation Engine (Step 9).

### 4. Compute and Return Outputs
Create a simulation tick/update cycle to return the delta for the following outputs:
- [ ] **Revenue Impact:** Calculate revenue changes based on marketing, product launch, and team scaling.
- [ ] **Funding Impact:** Calculate burn rate updates after hiring and marketing changes.
- [ ] **Growth Impact:** Model user/customer growth based on actions taken.
- [ ] **Risk Impact:** 
  - Call the `failure_prediction_model` using the newly updated `team_size`, `months_since_launch`, and `text`.
  - Output the delta in the `risk_indicator` (0-100 score).

### 5. Integration and Testing
- [ ] Write unit tests passing a dummy startup through sequential actions.
- [ ] Verify that predicting failure risk directly scales with hiring rate and launch time (testing the contemporaneous effects discussed in the model's limitations).
- [ ] Feed the Twin Outputs back into the VENTUREGENESIS agent decisions loop.