if __package__ in [None, '']:
    import MeterTransformerPairing
    import OnlinePhaseChangePoint
    import PhaseIdentification
else:
    from . import MeterTransformerPairing
    from . import OnlinePhaseChangePoint
    from . import PhaseIdentification

def meter_transformer_pairing(in_csv,out_csv):
    pass #TODO: implement by calling the correct submodule function

def online_phase_change_point(in_csv,out_csv):
    pass #TODO: implement by calling the correct submodule function

def phase_identification(in_csv,out_csv):
    pass #TODO: implement by calling the correct submodule function

def _run_all_tests():
    pass #TODO: maybe add code to test each of the submodules here? I.e. just call some of the functions in the submodules on the existing sample data to make sure they don't crash.