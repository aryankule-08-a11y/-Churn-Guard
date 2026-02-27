"""
📊 Sample Customer Churn Dataset Generator
Generates a realistic telecom customer churn dataset for demo purposes.
"""

import pandas as pd
import numpy as np

def generate_sample_dataset(n_samples=1000):
    """Generate a realistic customer churn dataset."""
    np.random.seed(42)
    
    genders = np.random.choice(['Male', 'Female'], n_samples)
    ages = np.random.randint(18, 72, n_samples)
    tenures = np.random.randint(0, 72, n_samples)
    monthly_charges = np.round(np.random.uniform(18.0, 118.0, n_samples), 2)
    contract_types = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.25, 0.25])
    internet_services = np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.20])
    payment_methods = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples
    )
    total_charges = np.round(monthly_charges * tenures + np.random.uniform(-50, 50, n_samples), 2)
    total_charges = np.maximum(total_charges, 0)

    # Generate churn labels based on realistic patterns
    churn_prob = np.zeros(n_samples)
    churn_prob += (contract_types == 'Month-to-month') * 0.25
    churn_prob += (internet_services == 'Fiber optic') * 0.15
    churn_prob += (payment_methods == 'Electronic check') * 0.10
    churn_prob += (tenures < 12) * 0.15
    churn_prob += (monthly_charges > 70) * 0.10
    churn_prob += np.random.uniform(-0.05, 0.05, n_samples)
    churn_prob = np.clip(churn_prob, 0.05, 0.95)
    churn = (np.random.rand(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        'customerID': [f'CUST-{i:04d}' for i in range(1, n_samples + 1)],
        'Gender': genders,
        'Age': ages,
        'Tenure': tenures,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'ContractType': contract_types,
        'InternetService': internet_services,
        'PaymentMethod': payment_methods,
        'Churn': churn
    })
    return df

if __name__ == '__main__':
    df = generate_sample_data(1000)
    df.to_csv('sample_churn_data.csv', index=False)
    print(f"✅ Generated {len(df)} rows. Churn rate: {df['Churn'].mean():.1%}")
