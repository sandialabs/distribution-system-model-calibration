if __package__ is None or __package__ == '':
    import M2TFuncs
    import M2TUtils
    import MeterToTransPairingScripts
    import MeterToTransPairingScript_WithDistance
else:
    from . import M2TFuncs
    from . import M2TUtils
    from . import MeterToTransPairingScripts
    from . import MeterToTransPairingScript_WithDistance
#TODO: probably don't need all these imports.