from dataclasses import fields
import json
from lcs.joint_learning import run_config, get_tape_module
import logging
logger = logging.getLogger(__name__)
from lcs import SRC_ROOT
from pathlib import Path
import equinox as eqx
import numpy as np
from lcs.configs import Config, cfg_from_string

logger = logging.getLogger(__name__)

def get_data(cfg=None, recompute=False, overrides=dict()):
    """
    Receives a Config datackass or a cli_string. 
    Checks if cached result exists and otherwise runs simulation and saves data.
    """
    if type(cfg) == str:
        cfg = cfg_from_string(cfg)
    elif type(cfg) == dict:
        cfg = Config(**cfg)
    else:
        pass

    from dataclasses import replace
    cfg = replace(cfg, **overrides)

    data_path = Path(cfg.data_out_dir) / cfg.name
    try:
        if recompute: raise FileNotFoundError  # hack
        with open(data_path / 'cfg.json', 'r') as f:
            args = json.load(f)
            cfg_ = Config(**args)
            tape_module = get_tape_module(cfg_, True)
            tape = eqx.tree_deserialise_leaves(data_path / 'tape.eqx', tape_module)
        if hash(cfg) != hash(cfg_): raise ValueError  # hack
        logger.info(f"Data found at {data_path}. Loaded data.")
    except (FileNotFoundError, ValueError, TypeError) as e:
        if type(e) == FileNotFoundError:
            logger.info(f"Data not found at {data_path}. Running simulation.")
        elif type(e) == ValueError:
            logger.warning(f"Data found at {data_path} but with different config. Re-running simulation.")
        elif type(e) == TypeError:
            logger.warning(f"{e}. Re-running simulation.")

        tape, W_teacher, cfg = run_config(cfg, return_teachers=True, return_cfg=True)

        # Saving data and config    
        data_path.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(data_path / 'tape.eqx', tape)
        with open(data_path / 'cfg.json', 'w') as f:
            # saves only the fields that are needed to uniquely reinitialize the config
            json.dump({field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init and not hasattr(getattr(cfg, field.name), 'shape')}, f) #, indent=2)  
        np.save(data_path / 'W_teacher.npy', W_teacher)

    return tape, cfg

if __name__ == "__main__":
    from lcs.configs import parser
    tape, cfg = get_data(cfg=parser.parse_args().config)

    # ---- Code for reloading saved tape and teacher (data) for offline plotting
    # data_path = Path(cfg.data_out_dir) / cfg.name
    # with open(data_path / 'cfg.json', 'r') as f:
    #     cfg = json.load(f)
    # cfg = Config(**cfg)
    # tape_module = get_tape_module(cfg, True)
    # tape = eqx.tree_deserialise_leaves(data_path / 'tape.eqx', tape_module)
    # W_teacher = np.load(data_path / 'W_teacher.npy')