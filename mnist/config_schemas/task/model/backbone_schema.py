from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING

@dataclass
class BackboneConfig:
    _target_:str = MISSING

@dataclass
class Resnet18BackboneConfig(BackboneConfig):
    _target_:str = "backbones.ResNet18"
    pretrained:bool = True

@dataclass
class Resnet34BackboneConfig(BackboneConfig):
    _target_:str = "backbones.ResNet34"
    pretrained:bool = True
    
@dataclass
class Resnet50BackboneConfig(BackboneConfig):
    _target_:str = "backbones.ResNet50"
    pretrained:bool = True

def setup_config()->None:
    cs = ConfigStore.instance()
    cs.store(group="task/model/backbone", name="resnet18_schema", node=Resnet18BackboneConfig)
    cs.store(group="task/model/backbone", name="resnet34_schema", node=Resnet34BackboneConfig)
    cs.store(group="task/model/backbone", name="resnet50_schema", node=Resnet50BackboneConfig)



