import hydra
from omegaconf import OmegaConf

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(config)->None:
    print(OmegaConf.to_yaml(config))

if __name__=="__main__":
    main()
