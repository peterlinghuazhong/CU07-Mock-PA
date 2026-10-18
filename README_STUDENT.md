# CU7 MLflow PA (MOCK – practice) — PinjamPantas  ·  how to start

1. Install the packages: `pip install -r requirements.txt` (MLflow **3.16.1**, the version used in Week 3).
2. Create **your own** data file (use the last four digits of your IC, keep the leading zero):
   `python make_my_data.py 0731`  →  creates `loan_applicants.csv` in this folder. Use this file in your notebook.
3. Open `PA_Starter.ipynb`, fill in your name and `L4`, and work through the tasks of the paper in order.
4. Your MLflow store is `pinjampantas_<L4>.db` (+ the `mlruns/` folder), created in the folder where you run the notebook.
5. When you finish, check yourself: `python check_my_work.py <L4>` (add `--serve` while your model server is running).
   Fix every [FAIL] and run the notebook again from a fresh store (delete the `.db` file and `mlruns/` first).

Tips: serve the model on **your own port** (5100 + the last two digits of your IC); on macOS port 5000 is already used by AirPlay.
