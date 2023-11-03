if __package__ in [None, '']:
    import ChangepointUtils
    import OnlineChangepoint_Script
    import CreateTimeDurationCurve_Script
else:
    from . import ChangepointUtils
    from . import OnlineChangepoint_Script
    from . import CreateTimeDurationCurve_Script

