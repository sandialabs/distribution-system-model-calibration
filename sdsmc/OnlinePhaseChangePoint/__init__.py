if __package__ in [None, '']:
    import ChangepointUtils
    import OnlineChangepoint_Script
    import CreateTimeDurationCurve_Script
    import CreateTimeDurationCurve
    import PhaseChangepoint
else:
    from . import ChangepointUtils
    from . import OnlineChangepoint_Script
    from . import CreateTimeDurationCurve_Script
    from . import CreateTimeDurationCurve
    from . import PhaseChangepoint

