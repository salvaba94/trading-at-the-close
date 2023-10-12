import json
import re
from pathlib import Path
from typing import List, Mapping, Any
import pickle

#==============================================================================

def read_json(
    file: Path
) -> Mapping[str, Any]:
    """
    This function reads a json file.
    """

    with open(file, "r") as f:
        content = f.read()
        content = re.sub("//.*", "", content, flags=re.MULTILINE)
        content = json.loads(content)

    return content

#==============================================================================

def dump_json(
    file: Path,
    content: Mapping[str, Any],
    indent: int = None
) -> None:
    """
    This function dumps the dictionary to file.
    """

    with open(file, "w") as f:
        json.dump(content, f, indent=indent)


#==============================================================================

def save_model(
        model: Any,
        file: Path
    ) -> None:
    """Saves the model to disk in {name}.txt and {name}.pkl format"""

    try:
        model.booster_.save_model(str(file.with_suffix(".txt"))) # type lgb.basic.Booster
    except:
        pass

    # Save the model with pickle
    with open(file.with_suffix(".pkl"), "wb") as f:
        pickle.dump(model, f)


#==============================================================================

def load_model(
        model, 
        file: Path
    ) -> Any:
    """Reads a model from a pickle file"""

    with open(file.with_suffix(".pkl"), "rb") as f:
        model = pickle.load(f)

    return model


#==============================================================================