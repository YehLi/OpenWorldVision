from .cross_entropy import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from .jsd import JsdCrossEntropy
from .asymmetric_loss import AsymmetricLossMultiLabel, AsymmetricLossSingleLabel
from .rpl_loss import RplLoss
from .min_unknown_rpl_loss import MinUnknownRplLoss
from .min_unknown_rpl_loss_plus import MinUnknownRplLossPlus
from .ce_centroids_loss import CECentroidsLoss
from .ce_centroids_loss2 import CECentroidsLoss2
from .ce_centroids_loss_nograd import CECentroidsLossNoGrad
from .ce_supmoco_loss import CESupMoCoLoss
from .general_entropy import GeneralEntropyLoss
from .active_passive_loss import NCEandRCE