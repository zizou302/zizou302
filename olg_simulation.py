import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')


class OLGSimulation:
    def __init__(self):
        """Initialize model parameters from Tables 1 and 2"""

        # Individual parameters
        self.rho = 0.04          # Annual discount rate
        self.sigma = 0.138       # Family's propensity to save
        self.epsilon = 0.123     # Time allocated to schooling
        self.mu = 0.15           # Training cost parameter
        self.kappa = 0.5         # Productivity parameter

        # Additional labor market wedges (new)
        # Baseline assumptions consistent with user's notes: mu_D baseline = 0.2
        # Others at 0 unless specified by reforms
        self.mu_D = 0.2          # Additional wedge in innovation/R&D labor market
        self.mu_Y = 0.0          # Additional wedge in final goods labor market
        self.mu_I = 0.0          # Additional wedge in imitation labor market

        # Final goods production
        self.omega = 0.24        # Elasticity w.r.t public-private capital ratio
        self.alpha = 0.35        # Elasticity w.r.t private capital
        self.beta_U = 0.2        # Elasticity w.r.t unskilled labor
        self.beta_S = 0.35       # Elasticity w.r.t skilled labor
        self.gamma = 0.15        # Elasticity w.r.t intermediate goods
        self.v = 0.9             # Share of core inputs
        self.phi_Y = 0.05        # Labor market distortion in final goods

        # Intermediate inputs
        self.eta = 0.25          # Substitution parameter
        self.n = self.eta        # For consistency with the model

        # Imitation sector
        self.kappa_I = 0.35      # Elasticity w.r.t technology frontier
        self.phi_1 = 0.2         # Elasticity w.r.t public infrastructure
        self.phi_2 = 0.0         # Productivity parameter
        self.g_I_W = 0.02        # Growth rate of world imitable goods
        self.phi_I = 0.1         # Labor market distortion in imitation

        # Innovation sector
        self.delta_1 = 0.3       # Elasticity w.r.t public infrastructure
        self.delta_2 = 0.0       # Productivity parameter
        self.delta_3 = 0.04      # Learning effect from imitation
        self.lambda_param = 0.8  # Property rights inefficiency
        self.phi_R = 0.2         # Labor market distortion in innovation

        # Government
        self.tau = 0.123         # Tax rate
        self.upsilon_I = 0.069   # Share of spending on infrastructure

        # Calculate derived parameters
        self.beta = self.beta_U / self.beta_S
        self.J_I = (self.upsilon_I * self.tau) / (self.sigma * (1 - self.tau))

        # Initial steady-state values
        self.theta_U_0 = 0.95
        self.theta_S_0 = 0.049
        self.theta_UI_0 = 0.25
        self.theta_UY_0 = 0.7
        self.theta_SR_0 = 0.009
        self.theta_SY_0 = 0.04
        self.m_I_0 = 0.05
        self.m_R_0 = 0.01

    def _recompute_derived(self):
        """Recompute derived parameters that depend on base params."""
        self.beta = self.beta_U / self.beta_S
        self.n = self.eta
        self.J_I = (self.upsilon_I * self.tau) / (self.sigma * (1 - self.tau))

    def calculate_ability_threshold(self, theta_SY, theta_UY):
        """Calculate ability threshold (equation 41)"""
        ratio = theta_SY / theta_UY
        a_c = self.mu**(1/self.kappa) * (1 - self.beta * ratio)**(-1/self.kappa)
        return min(max(a_c, 0.001), 0.999)  # Bound between 0 and 1

    def calculate_labor_shares(self, a_c):
        """Calculate labor shares (equations 41-42)"""
        theta_U = a_c
        theta_S = (1 - a_c**2) / 2
        return theta_U, theta_S

    def calculate_theta_SY(self, m_R, m_I):
        """Calculate skilled labor share in final goods (equation 43)"""
        numerator = self.beta_S * (1 + self.phi_R + self.mu_D) * (self.J_I**(-self.delta_1))
        denominator = (self.gamma * (1 - self.v) * (1 - self.n) *
                       (1 - self.epsilon) * (1 - self.lambda_param) * (1 + self.phi_Y + self.mu_Y))
        stepping_stone = 1 + self.delta_3 * (m_I / m_R) if m_R > 0 else 1

        theta_SY = (numerator / denominator) * (m_R**(-self.delta_2)) / stepping_stone
        return min(theta_SY, 0.5)  # Cap at reasonable level

    def calculate_theta_UY(self, m_I):
        """Calculate unskilled labor share in final goods (equation 45)"""
        numerator = self.beta_U * (1 + self.phi_I + self.mu_I) * (self.J_I**(-self.phi_1))
        denominator = (self.gamma * self.v * (1 - self.n) * (1 + self.phi_Y + self.mu_Y) *
                       ((1 + self.g_I_W)**self.kappa_I))

        theta_UY = (numerator / denominator) * (m_I**(-self.phi_2))
        return min(theta_UY, 0.9)  # Cap at reasonable level

    def calculate_output_capital_ratio(self, theta_SY, theta_UY, m_I, m_R):
        """Calculate Y/K^P ratio (equation 38)"""
        coeff = ((self.gamma**2 * self.n**2 * self.v**self.v *
                 (1-self.v)**(1-self.v))**(self.gamma/(1-self.gamma)) *
                (1-self.epsilon)**(self.beta_S/(1-self.gamma)) *
                (self.J_I)**(self.omega/(1-self.gamma)))

        labor_term = (theta_SY**self.beta_S * theta_UY**self.beta_U)**(-1/(1-self.gamma))
        knowledge_term = ((m_I**((1-self.n)*self.v/self.n) *
                          m_R**((1-self.n)*(1-self.v)/self.n))**(self.gamma/(1-self.gamma)))

        return coeff * labor_term * knowledge_term

    def calculate_patent_prices(self, Y_KP, m_I, m_R):
        """Calculate patent prices (equations 39-40)"""
        base_R = (self.gamma * (1 - self.v) * (1 - self.n) * Y_KP)
        F_R = base_R / m_R if m_R > 0 else 0.1

        base_I = (self.gamma * self.v * (1 - self.n) * Y_KP)
        F_I = base_I / m_I if m_I > 0 else 0.1

        return F_R, F_I

    def update_knowledge_ratios(self, m_I, m_R, theta_U, theta_S, theta_UI, theta_SR, F_I, F_R):
        """Update knowledge-capital ratios (equations 36-37)"""
        # Growth in imitation designs
        growth_I = ((self.J_I**self.phi_1) * ((1 + self.g_I_W)**self.kappa_I) *
                   (m_I**self.phi_2) * theta_UI)

        # Growth in innovation designs
        stepping_stone = 1 + self.delta_3 * (m_I / m_R) if m_R > 0 else 1
        growth_R = ((1 - self.epsilon) * (self.J_I**self.delta_1) *
                   (m_R**self.delta_2) * stepping_stone * theta_SR)

        # Common denominator for capital accumulation
        term1 = ((self.sigma * (1 - self.tau) * ((1 + self.g_I_W)**self.kappa_I) *
                 (self.J_I**self.phi_1)) / (1 + self.phi_I + self.mu_I)) * F_I * theta_U * (m_I**(self.phi_2 + 1))

        avg_ability = 0.5 * (1 + theta_U)**self.kappa if theta_U < 1 else 1
        term2_coeff = (1 - self.mu / avg_ability) * ((self.sigma * (1 - self.tau) *
                      (1 - self.epsilon) * (1 - self.lambda_param) * (self.J_I**self.delta_1)) /
                      (1 + self.phi_R + self.mu_D))
        term2 = term2_coeff * F_R * theta_S * (m_R**(self.delta_2 + 1)) * stepping_stone

        capital_growth = 1 / (term1 + term2) if (term1 + term2) > 0 else 1

        # Update ratios
        m_I_new = m_I * (1 + growth_I) * capital_growth
        m_R_new = m_R * (1 + growth_R) * capital_growth

        return m_I_new, m_R_new

    def simulate(self, T=100, shock_period=None, shock_type=None, shock_magnitude=0.1, params_override=None):
        """
        Run simulation for T periods
        shock_period: period when shock occurs
        shock_type: 'phi_I', 'phi_R', 'phi_Y', 'lambda', 'g_I_W', 'J_I'
        shock_magnitude: size of shock (proportional change)
        params_override: dict of parameter -> value to temporarily set for this run
        """
        # Store and apply parameter overrides if provided
        overridden = {}
        if params_override:
            for key, value in params_override.items():
                if not hasattr(self, key):
                    continue
                overridden[key] = getattr(self, key)
                setattr(self, key, value)
            # Some overrides affect derived params
            self._recompute_derived()

        # Initialize storage arrays
        results = {
            'm_I': np.zeros(T),
            'm_R': np.zeros(T),
            'theta_U': np.zeros(T),
            'theta_S': np.zeros(T),
            'theta_UY': np.zeros(T),
            'theta_SY': np.zeros(T),
            'theta_UI': np.zeros(T),
            'theta_SR': np.zeros(T),
            'Y_KP': np.zeros(T),
            'F_I': np.zeros(T),
            'F_R': np.zeros(T),
            'industrial_transformation': np.zeros(T),
            'growth_rate': np.zeros(T),
            'wage_ratio': np.zeros(T)
        }

        # Set initial values
        results['m_I'][0] = self.m_I_0
        results['m_R'][0] = self.m_R_0
        results['theta_U'][0] = self.theta_U_0
        results['theta_S'][0] = self.theta_S_0
        results['theta_UY'][0] = self.theta_UY_0
        results['theta_SY'][0] = self.theta_SY_0
        results['theta_UI'][0] = self.theta_UI_0
        results['theta_SR'][0] = self.theta_SR_0

        # Store original parameter values for shocks
        original_params = {
            'phi_I': self.phi_I,
            'phi_R': self.phi_R,
            'phi_Y': self.phi_Y,
            'lambda': self.lambda_param,
            'g_I_W': self.g_I_W,
            'J_I': self.J_I
        }

        # Main simulation loop
        for t in range(T):
            # Apply shock if specified
            if shock_period is not None and t >= shock_period and shock_type is not None:
                if shock_type == 'phi_I':
                    self.phi_I = original_params['phi_I'] * (1 - shock_magnitude)
                elif shock_type == 'phi_R':
                    self.phi_R = original_params['phi_R'] * (1 - shock_magnitude)
                elif shock_type == 'phi_Y':
                    self.phi_Y = original_params['phi_Y'] * (1 - shock_magnitude)
                elif shock_type == 'lambda':
                    self.lambda_param = original_params['lambda'] * (1 - shock_magnitude)
                elif shock_type == 'g_I_W':
                    self.g_I_W = original_params['g_I_W'] * (1 + shock_magnitude)
                elif shock_type == 'J_I':
                    self.J_I = original_params['J_I'] * (1 + shock_magnitude)

            # Current state
            m_I = results['m_I'][t]
            m_R = results['m_R'][t]

            # Calculate labor allocations
            theta_UY = self.calculate_theta_UY(m_I)
            theta_SY = self.calculate_theta_SY(m_R, m_I)

            # Calculate ability threshold and labor supplies
            a_c = self.calculate_ability_threshold(theta_SY, theta_UY)
            theta_U, theta_S = self.calculate_labor_shares(a_c)

            # Calculate sectoral allocations
            theta_UI = max(0, theta_U - theta_UY)
            theta_SR = max(0, theta_S - theta_SY)

            # Calculate output and prices
            Y_KP = self.calculate_output_capital_ratio(theta_SY, theta_UY, m_I, m_R)
            F_R, F_I = self.calculate_patent_prices(Y_KP, m_I, m_R)

            # Store results
            results['theta_U'][t] = theta_U
            results['theta_S'][t] = theta_S
            results['theta_UY'][t] = theta_UY
            results['theta_SY'][t] = theta_SY
            results['theta_UI'][t] = theta_UI
            results['theta_SR'][t] = theta_SR
            results['Y_KP'][t] = Y_KP
            results['F_I'][t] = F_I
            results['F_R'][t] = F_R

            # Calculate additional indicators
            results['industrial_transformation'][t] = m_I / (m_I + m_R) if (m_I + m_R) > 0 else 0
            results['wage_ratio'][t] = self.beta * (1 - self.epsilon) * theta_SY / theta_UY if theta_UY > 0 else 0

            # Update for next period (if not last period)
            if t < T - 1:
                m_I_new, m_R_new = self.update_knowledge_ratios(
                    m_I, m_R, theta_U, theta_S, theta_UI, theta_SR, F_I, F_R
                )
                results['m_I'][t + 1] = m_I_new
                results['m_R'][t + 1] = m_R_new

                # Calculate growth rate
                if t > 0:
                    results['growth_rate'][t] = (Y_KP - results['Y_KP'][t-1]) / results['Y_KP'][t-1] if results['Y_KP'][t-1] > 0 else 0

        # Restore overrides
        if overridden:
            for key, old_val in overridden.items():
                setattr(self, key, old_val)
            self._recompute_derived()

        return results

    def plot_results(self, results, title_suffix=""):
        """Create visualization of simulation results"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'OLG Model Simulation: Imitation to Innovation Dynamics {title_suffix}', fontsize=14, fontweight='bold')

        time = np.arange(len(results['m_I']))

        # Plot 1: Knowledge-capital ratios
        axes[0, 0].plot(time, results['m_I'], label='m_I (Imitation)', color='blue', linewidth=2)
        axes[0, 0].plot(time, results['m_R'], label='m_R (Innovation)', color='red', linewidth=2)
        axes[0, 0].set_xlabel('Period')
        axes[0, 0].set_ylabel('Knowledge-Capital Ratio')
        axes[0, 0].set_title('Evolution of Knowledge Stocks')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Labor shares in population
        axes[0, 1].plot(time, results['theta_U'], label='Unskilled (θ_U)', color='orange', linewidth=2)
        axes[0, 1].plot(time, results['theta_S'], label='Skilled (θ_S)', color='green', linewidth=2)
        axes[0, 1].set_xlabel('Period')
        axes[0, 1].set_ylabel('Share of Population')
        axes[0, 1].set_title('Labor Force Composition')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Industrial transformation index
        axes[0, 2].plot(time, results['industrial_transformation'], color='purple', linewidth=2)
        axes[0, 2].set_xlabel('Period')
        axes[0, 2].set_ylabel('Transformation Index')
        axes[0, 2].set_title('Industrial Transformation (m_I/(m_I+m_R)')
        axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Sectoral allocation of unskilled labor
        axes[1, 0].plot(time, results['theta_UY'], label='Final Goods (θ_UY)', color='blue', linewidth=2)
        axes[1, 0].plot(time, results['theta_UI'], label='Imitation (θ_UI)', color='cyan', linewidth=2)
        axes[1, 0].set_xlabel('Period')
        axes[1, 0].set_ylabel('Share')
        axes[1, 0].set_title('Unskilled Labor Allocation')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Sectoral allocation of skilled labor
        axes[1, 1].plot(time, results['theta_SY'], label='Final Goods (θ_SY)', color='red', linewidth=2)
        axes[1, 1].plot(time, results['theta_SR'], label='Innovation (θ_SR)', color='darkred', linewidth=2)
        axes[1, 1].set_xlabel('Period')
        axes[1, 1].set_ylabel('Share')
        axes[1, 1].set_title('Skilled Labor Allocation')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Output-capital ratio
        axes[1, 2].plot(time, results['Y_KP'], color='green', linewidth=2)
        axes[1, 2].set_xlabel('Period')
        axes[1, 2].set_ylabel('Y/K^P')
        axes[1, 2].set_title('Output-Capital Ratio')
        axes[1, 2].grid(True, alpha=0.3)

        # Plot 7: Patent prices
        axes[2, 0].plot(time, results['F_I'], label='Imitation (F_I)', color='blue', linewidth=2)
        axes[2, 0].plot(time, results['F_R'], label='Innovation (F_R)', color='red', linewidth=2)
        axes[2, 0].set_xlabel('Period')
        axes[2, 0].set_ylabel('Patent Price')
        axes[2, 0].set_title('Patent Prices')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # Plot 8: Growth rate
        axes[2, 1].plot(time[1:], results['growth_rate'][1:], color='darkgreen', linewidth=2)
        axes[2, 1].set_xlabel('Period')
        axes[2, 1].set_ylabel('Growth Rate')
        axes[2, 1].set_title('Economic Growth Rate')
        axes[2, 1].grid(True, alpha=0.3)

        # Plot 9: Wage ratio
        axes[2, 2].plot(time, results['wage_ratio'], color='brown', linewidth=2)
        axes[2, 2].set_xlabel('Period')
        axes[2, 2].set_ylabel('w_U/w_S')
        axes[2, 2].set_title('Unskilled-Skilled Wage Ratio')
        axes[2, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def run_policy_experiments(self):
        """Run original policy experiments (kept for backward compatibility)."""
        experiments = {
            'Baseline': (None, None, 0),
            'Reduce Innovation Labor Distortion': ('phi_R', 0.0, 50),
            'Reduce Imitation Labor Distortion': ('phi_I', 0.5, 50),
            'Improve Property Rights': ('lambda', 0.1, 50),
            'Increase Infrastructure': ('J_I', 0.2, 50)
        }

        results_dict = {}

        for name, (shock_type, magnitude, period) in experiments.items():
            print(f"Running experiment: {name}")
            if shock_type:
                results = self.simulate(T=100, shock_period=period,
                                      shock_type=shock_type, shock_magnitude=magnitude)
            else:
                results = self.simulate(T=100)
            results_dict[name] = results

        return results_dict

    def run_reform_experiments(self):
        """Run new reform scenarios specified by the user."""
        scenarios = {}

        # Benchmark
        scenarios['Benchmark'] = ({},)

        # 1) Skill expansion under Benchmark economy
        scenarios['Skill Expansion'] = ({'mu': 0.1, 'mu_D': 0.0, 'epsilon': 0.15},)

        # 2) Labor market reforms (7 cases)
        labor_reforms = [
            ('Labor Reform 1', {'mu_D': 0.0, 'mu_Y': 0.05, 'mu_I': 0.1}),
            ('Labor Reform 2', {'mu_D': 0.0, 'mu_Y': 0.2,  'mu_I': 0.1}),
            ('Labor Reform 3', {'mu_D': 0.0, 'mu_Y': 0.2,  'mu_I': 0.2}),
            ('Labor Reform 4', {'mu_D': 0.0, 'mu_Y': 0.2,  'mu_I': 0.0}),
            ('Labor Reform 5', {'mu_D': 0.0, 'mu_Y': 0.0,  'mu_I': 0.2}),
            ('Labor Reform 6', {'mu_D': 0.0, 'mu_Y': 0.05, 'mu_I': 0.2}),
            ('Labor Reform 7', {'mu_D': 0.3, 'mu_Y': 0.05, 'mu_I': 0.1}),
        ]
        for name, params in labor_reforms:
            scenarios[name] = (params,)

        # 3) Enforcement of property rights
        prop_rights = [
            ('Property Rights 1', {'lambda_param': 0.4, 'eta': 0.41, 'delta_2': 0.2}),
            ('Property Rights 2', {'lambda_param': 0.1, 'eta': 0.41, 'delta_2': 0.3}),
            ('Property Rights 3', {'lambda_param': 0.0, 'eta': 0.41, 'delta_2': 0.4}),
        ]
        for name, params in prop_rights:
            scenarios[name] = (params,)

        # 4) Increase in public infrastructure
        infrastructure = [
            ('Infrastructure 1', {'upsilon_I': 0.10, 'lambda_param': 0.1, 'eta': 0.41, 'delta_2': 0.4}),
            ('Infrastructure 2', {'upsilon_I': 0.15, 'lambda_param': 0.1, 'eta': 0.41, 'delta_2': 0.4}),
            ('Infrastructure 3', {'upsilon_I': 0.17, 'lambda_param': 0.1, 'eta': 0.41, 'delta_2': 0.4}),
        ]
        for name, params in infrastructure:
            scenarios[name] = (params,)

        results = {}
        for name, (params,) in scenarios.items():
            print(f"Running scenario: {name}")
            # Recompute derived inside simulate via params_override
            res = self.simulate(T=100, params_override=params)
            results[name] = res

        return results

    @staticmethod
    def _collect_metrics(results):
        """Collect final period metrics needed for comparison table."""
        idx = -1
        return {
            'Industrial transformation': results['industrial_transformation'][idx],
            'Final output': results['Y_KP'][idx],
            'Unskilled labor': results['theta_U'][idx],
            'Skilled labor': results['theta_S'][idx],
            'Unskilled in imitation': results['theta_UI'][idx],
            'Skilled in innovation': results['theta_SR'][idx],
        }

    @staticmethod
    def _percent_change(new, base):
        if base == 0:
            return np.nan
        return 100.0 * (new - base) / base

    def build_comparison_table(self, scenarios_results):
        """Build a DataFrame of percent impacts relative to Benchmark."""
        if 'Benchmark' not in scenarios_results:
            raise ValueError('Benchmark scenario missing')

        base_metrics = self._collect_metrics(scenarios_results['Benchmark'])

        rows = []
        for name, res in scenarios_results.items():
            if name == 'Benchmark':
                continue
            m = self._collect_metrics(res)
            row = {
                'Parameter / Reform': name,
                'Industrial transformation': self._percent_change(m['Industrial transformation'], base_metrics['Industrial transformation']),
                'Final output': self._percent_change(m['Final output'], base_metrics['Final output']),
                'Unskilled labor': self._percent_change(m['Unskilled labor'], base_metrics['Unskilled labor']),
                'Skilled labor': self._percent_change(m['Skilled labor'], base_metrics['Skilled labor']),
                'Unskilled in imitation': self._percent_change(m['Unskilled in imitation'], base_metrics['Unskilled in imitation']),
                'Skilled in innovation': self._percent_change(m['Skilled in innovation'], base_metrics['Skilled in innovation']),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        # Order columns as requested
        df = df[[
            'Parameter / Reform',
            'Industrial transformation',
            'Final output',
            'Unskilled labor',
            'Skilled labor',
            'Unskilled in imitation',
            'Skilled in innovation',
        ]]
        return df


def main():
    print("Initializing OLG Model Simulation...")
    model = OLGSimulation()

    # Run baseline simulation
    print("\nRunning baseline simulation...")
    baseline_results = model.simulate(T=100)

    # Create visualization
    print("Creating baseline visualization...")
    fig1 = model.plot_results(baseline_results, title_suffix="(Baseline)")
    plt.show()

    # Run original policy experiments
    print("\nRunning original policy experiments...")
    experiments = model.run_policy_experiments()

    # Compare experiments
    print("Comparing original policy experiments...")
    fig2 = model.compare_experiments(experiments) if hasattr(model, 'compare_experiments') else None
    if fig2 is not None:
        plt.show()

    # Run reform experiments per user request
    print("\nRunning reform experiments...")
    reform_results = model.run_reform_experiments()

    # Build and print comparison table vs Benchmark
    print("\nBuilding comparison table vs Benchmark...")
    table = model.build_comparison_table(reform_results)
    # Pretty print
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print(table.to_string(index=False, justify='left', float_format=lambda x: f"{x:+.2f}%"))

    # Summary statistics for baseline
    print("\n" + "="*60)
    print("SIMULATION SUMMARY STATISTICS")
    print("="*60)

    print("\nBaseline Steady-State Values (t=99):")
    print(f"Industrial Transformation Index: {baseline_results['industrial_transformation'][-1]:.4f}")
    print(f"Skilled Labor Share: {baseline_results['theta_S'][-1]:.4f}")
    print(f"Innovation Knowledge Ratio (m_R): {baseline_results['m_R'][-1]:.4f}")
    print(f"Imitation Knowledge Ratio (m_I): {baseline_results['m_I'][-1]:.4f}")
    print(f"Output-Capital Ratio: {baseline_results['Y_KP'][-1]:.4f}")
    print(f"Wage Ratio (w_U/w_S): {baseline_results['wage_ratio'][-1]:.4f}")

    print("\n" + "="*60)
    print("Simulation Complete!")


if __name__ == '__main__':
    main()

