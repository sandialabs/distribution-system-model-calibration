if __package__ in [None, '']:
    import CA_Ensemble_Funcs
    import CA_Ensemble_SampleScripts
    import PhaseIdent_Utils
    import SensorMethod_Funcs
    import SensorMethod_SampleScript
else:
    from . import CA_Ensemble_Funcs
    from . import CA_Ensemble_SampleScripts
    from . import PhaseIdent_Utils
    from . import SensorMethod_Funcs
    from . import SensorMethod_SampleScript