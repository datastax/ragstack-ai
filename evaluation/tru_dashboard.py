import tru_shared

tru = tru_shared.initTru()
tru.run_dashboard(address="0.0.0.0", port=8501, force=True)
