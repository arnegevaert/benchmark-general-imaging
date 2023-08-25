# Baseline pseudo-methods
from .random import *
from .edge_detection import *

# Methods
from .deconvolution import Deconvolution
from .expected_gradients import ExpectedGradients
from .gradcam import GradCAM
from .gradient import Gradient
from .guided_backprop import GuidedBackprop
from .guided_gradcam import GuidedGradCAM
from .input_x_gradient import InputXGradient
from .integrated_gradients import IntegratedGradients
from .smooth_grad import SmoothGrad
from .var_grad import VarGrad
from .deeplift import DeepLift
from .kernel_shap import KernelShap
from .image_lime import ImageLime
from .tabular_lime import TabularLime
from .deep_shap import DeepShap
from .igos import Igos, Igos_pp
from .xrai import XRAI
from .extremal_perturbation import ExtremalPerturbation
from .gradcam_pp import GradCAMPP
from .scorecam import ScoreCAM
from .cgi import CGI

# Post-processing wrappers
from .pixel_aggregation import *  # Aggregate along color channels
