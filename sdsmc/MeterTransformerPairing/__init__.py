if __package__ in [None, '']:
    import M2TFuncs
    import M2TUtils
    import TransformerPairing
else:
    from . import M2TFuncs
    from . import M2TUtils
    from . import TransformerPairing