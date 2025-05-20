from names import GlobNames as gn

from models import model
from modules import module
from utils import shape

from models import multimodel
from models import barmodel
from models import dirimodel
from models import neumannmodel
from models import poissonmodel
from models import elasticmodel
from models import solidmodel
from models import timoshenkomodel
from models import framemodel
from models import linbcmodel
from models import trussmodel
from models import pointmassmodel
from models import diffusionmodel

from modules import initmodule
from modules import solvermodule
from modules import nonlinmodule
from modules import arclenmodule
from modules import outputmodule
from modules import vtkoutmodule
from modules import linbuckmodule
from modules import viewmodule
from modules import frameviewmodule
from modules import loaddispmodule
from modules import homogenizationmodule
from modules import graphmodule
from modules import modeshapemodule
from modules import explicittimemodule
from modules import newmarkmodule
from modules import nlnewmarkmodule
from modules import trapezoidalmodule
from modules import accelerationmodule

from utils import paramshapes

def declare_all(globdat):
    declare_models(globdat)
    declare_modules(globdat)
    declare_shapes(globdat)

def declare_models(globdat):
    factory = model.ModelFactory()

    multimodel.declare(factory)
    barmodel.declare(factory)
    dirimodel.declare(factory)
    neumannmodel.declare(factory)
    poissonmodel.declare(factory)
    elasticmodel.declare(factory)
    solidmodel.declare(factory)
    timoshenkomodel.declare(factory)
    framemodel.declare(factory)
    linbcmodel.declare(factory)
    trussmodel.declare(factory)
    pointmassmodel.declare(factory)
    diffusionmodel.declare(factory)

    globdat[gn.MODELFACTORY] = factory


def declare_modules(globdat):
    factory = module.ModuleFactory()

    initmodule.declare(factory)
    solvermodule.declare(factory)
    nonlinmodule.declare(factory)
    arclenmodule.declare(factory)
    outputmodule.declare(factory)
    vtkoutmodule.declare(factory)
    linbuckmodule.declare(factory)
    viewmodule.declare(factory)
    frameviewmodule.declare(factory)
    loaddispmodule.declare(factory)
    homogenizationmodule.declare(factory)
    graphmodule.declare(factory)
    modeshapemodule.declare(factory)
    explicittimemodule.declare(factory)
    newmarkmodule.declare(factory)
    nlnewmarkmodule.declare(factory)
    trapezoidalmodule.declare(factory)
    accelerationmodule.declare(factory)

    globdat[gn.MODULEFACTORY] = factory


def declare_shapes(globdat):
    factory = shape.ShapeFactory()

    paramshapes.declare(factory)

    globdat[gn.SHAPEFACTORY] = factory
