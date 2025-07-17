# Aave-V2-Wallet-Credit-Scoring-Model
developed a robust machine learning model that assigns a credit score between 0 and 1000 to each wallet, based solely on historical transaction behavior. Higher scores indicate reliable and responsible usage; lower scores reflect risky, bot-like, or exploitative behavior.
Model Overview
This machine learning model generates credit scores (0-1000) for Aave V2 wallets based solely on transaction history. Higher scores indicate reliable, responsible protocol usage, while lower scores reflect risky, bot-like, or exploitative behavior.

Key Features & Financial Logic
The model evaluates wallets using these engineered features:

# 1. Repayment Health (35% weight)
Repayment Ratio: Σ(repay amounts) / Σ(borrow amounts)

Scores penalized when ratio < 0.9

Heavy penalty (50% score reduction) when ratio < 0.5

Rationale: Measures debt responsibility - core to creditworthiness

# 2. Liquidation Risk (25% weight)
Liquidation Events: Count of liquidationcall actions

Any liquidation event caps score at 300

Rationale: Liquidations indicate collateral failures - strongest risk signal

# 3. Leverage Behavior (20% weight)
Borrow/Deposit Ratio: borrow_count / deposit_count

Scores capped at 500 when ratio > 0.8

30% penalty when ratio > 3

Rationale: High borrowing relative to deposits indicates over-leverage

# 4. Transaction Patterns (15% weight)
Transaction Frequency: Transactions/day

Time Deviation: Standard deviation between transactions (seconds)

Low time deviation + high frequency → bot-like penalty

Rationale: Detects non-human/automated behavior

# 5. Action Balance (5% weight)
Action Ratios: % distribution of:

deposit

borrow

repay

redeemunderlying

liquidationcall

Rationale: Balanced deposits/repayments indicate healthy usage

Algorithm Architecture

<img width="793" height="1770" alt="deepseek_mermaid_20250717_787f32" src="https://github.com/user-attachments/assets/43978a4f-4500-45b2-947e-6f379cc36fb0" />

Modeling Approach
Unsupervised Anomaly Detection

Uses Isolation Forest algorithm

Detects abnormal behavior without labeled data

Naturally handles DeFi's adversarial environment

Score Conversion

Raw anomaly scores mapped to 0-1000 range:
credit_score = 1000 * (anomaly_score - min_score) / (max_score - min_score)

Rule-Based Adjustments

Incorporates domain knowledge:

python
if liquidation_events > 0:
    score = min(score, 300)
if repayment_ratio < 0.5:
    score *= 0.5
if borrow_deposit_ratio > 3:
    score *= 0.7

Validation Methodology
The model was validated using:

Synthetic Wallet Profiles:

Responsible User: Regular deposits, timely repayments → Scores 750+

Over-Leveraged: High borrowing, few deposits → Scores 300-500

Liquidated User: At least 1 liquidation → Score 300

Bot: High-frequency, low-deviation txns → Scores < 400

Distribution Analysis:

Expected distribution: Bell curve with right skew

Top 10% scores > 800

Bottom 5% scores < 200

Feature Importance:
<img width="2111" height="1350" alt="deepseek_mermaid_20250717_06cb77" src="https://github.com/user-attachments/assets/6426196a-5141-4d23-8606-41cc0cec5d86" />

# Extensibility & Customization
The model supports these enhancements:

# 1. Protocol Upgrades
python
For Aave V3
V3_ACTIONS = ['supply', 'borrow', 'repay', 'withdraw', 'liquidation']

# 2. New Features
Collateral Health Factor:

python
health_factor = collateral_value / debt_value
Asset Volatility Adjustment:

python
if volatile_asset_ratio > 0.7:
    score *= 0.8
# 3. Model Improvements
Supervised Fine-tuning:

python
if labeled_data_available:
    use GradientBoostingClassifier()
On-Chain Context:

python
features['defi_health_score'] = get_defi_composite_score(wallet)
Usage Instructions
Dependencies
bash
pip install pandas numpy scikit-learn
Generate Scores
bash
python aave_credit_scoring.py
Input
user-wallet-transactions.json format:

json
[
  {
    "userWallet": "0x...",
    "action": "deposit",
    "timestamp": 1629178166,
    "actionData": {
      "amount": "2000000000",
      "assetSymbol": "USDC",
      "assetPriceUSD": "1.00"
    }
  }
]
Output
wallet_credit_scores.json:

json
[
  {"user": "0x...", "credit_score": 850},
  {"user": "0x...", "credit_score": 320}
]
Limitations & Ethical Considerations
Data Limitations:

Single-protocol view

No off-chain identity data

Historical price approximations

Transparency:

Model explains low scores through features

Users can improve behavior to increase scores

Bias Mitigation:

No demographic or geographic features

Purely behavior-based scoring

This model provides a foundational credit scoring system for DeFi that can evolve with the ecosystem. Its transparent design enables continuous improvement while maintaining fairness and interpretability.
