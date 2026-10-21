# SupplyPulse AI – Control Tower

AI-driven supply-chain decision support (NED University PGD, Data Science and AI).

**What it does**
- Approval queue: approve / modify / reject the AI's replenishment recommendations, with a full audit trail and purchase orders.
- Reverse a decision: cancels the purchase order and returns the item to the queue (nothing is deleted).
- **Use your own data**: upload any store's sales file (CSV/Excel), match the columns, and get forecasts (1/7/14/30 days), stockout risk, order advice and suspicious-bill checks. The model is trained on the uploaded store's own data.

**Try it:** open the *Use your own data* tab and upload a file from `test_stores/`.

Files: `app.py` (main app), `store_ui.py` + `store_pipeline.py` (own-data page and engine), `week8_replenishment_recommendations.csv` and `supplypulse.db` (project data, SQLite).

Note: decisions saved in `supplypulse.db` on the hosted app reset when the app restarts. The data is simulated.
