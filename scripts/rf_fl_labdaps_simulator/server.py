import rf_fl_labdaps_simulator

executor = rf_fl_labdaps_simulator.RandomForestFlServer(session_id = 550, n_clients=21, n_estimators=550)

executor.run()
