from losses.cross_entropy import CELoss
from losses.mares_loss import MaresLoss
from losses.beta_mares_loss import BetaMaresLoss
from losses.mares_plus_loss import MAResPlusLoss
from losses.multi_loss import MultiLoss

def get_loss_function(name):
    name = name.lower()
    if name == 'ce' or name == 'crossentropy':
        return CELoss()
    elif name == 'mares':
        return MaresLoss()
    elif name == 'beta_mares':
        return BetaMaresLoss()
    elif name == 'mares_plus':
        return MAResPlusLoss()
    elif name == 'multi':
        return MultiLoss()
    else:
        raise ValueError(f"Unknown loss function: {name}")
