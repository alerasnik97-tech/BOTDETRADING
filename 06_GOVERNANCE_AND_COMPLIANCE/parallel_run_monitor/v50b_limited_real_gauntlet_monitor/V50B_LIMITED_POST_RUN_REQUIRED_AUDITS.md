# V50B LIMITED REAL GAUNTLET — POST-RUN REQUIRED AUDITS

The following audits must be executed IMMEDIATELY after the runner finishes:

1. **Family Coverage Audit**: Verify that each family (F06, F08, F12) has a representative number of trades.
2. **Temporal Concentration Audit**: Check for excessive trade clustering in specific days/hours.
3. **Max Trades/Day Audit**: Ensure the throttler successfully limited activity per config.
4. **Duplicate Tradeset Audit**: Check if different configs produced byte-identical trade lists.
5. **Parameter Collision Audit**: Cross-verify configs with identical parameters but different IDs.
6. **Throttler Contamination Audit**: Verify engine state isolation between sequential config runs.
7. **Engine Proof Verification**: Link every trade to a unique Engine Instance ID.
