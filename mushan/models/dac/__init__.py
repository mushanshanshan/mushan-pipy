import dac

def get_dac(model_type):
    return dac.DAC.load(dac.utils.download(model_type))