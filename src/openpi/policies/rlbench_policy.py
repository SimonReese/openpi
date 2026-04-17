import dataclasses
from dis import Instruction

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class RLBenchInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    At the end of the day, the model will take a dictionary as input during inference and training.
    How we build this dictionary can be different between environments (i.e. Libero and RLBench provides observations in different ways).
    Some environments may return an alredy generated dictionary of obs.
    Anyway there is no specific format of dict across environments. Therefore this class is used to convert the dictionary from the specific environment
    into a dict of standard format, that the network will be able to process.
    Also, this class can be used to perform transformations on the data, like converting in delta of actions or changing images representations.
    It all depends on the envirnoment used.
    (Well in rlbench we have the Observation class storing the elements, so either we create a dict manually in a format we like, or we convert the object into dict)
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.

        # For now we assume following format
        # data = {
        #     "observation/image" : np.ndarray,
        #     "observation/wrist_image" : np.ndarray,
        #     "observation/state" : np.ndarray,
        #     "actions" : np.ndarray,   # <--- those will be present only during training
        #     "instruction" : str
        # }
        # Images from RLBench should already be in uint8, but LeRobot convertss them into float32 (c, h, w)
        front_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        # Pad any non-existent images with zero-arrays of the appropriate shape.
        third_image =  np.zeros_like(front_image)   # TODO: well right now we only use 2 images, but the model requires 3. Pad to 0
        state = data["observation/state"]
        instruction = data["instruction"]


        # Create inputs dict. Do not change the keys in the dict below, since those are retrieved by the model
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": front_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": third_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset. TODO: no idea on this
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
            "prompt" : instruction  # Pass the prompt (aka language instruction) to the model. The output dict always needs to have the key "prompt"
        }

        # Pad actions to the model action dimension.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        return inputs
    
@dataclasses.dataclass(frozen=True)
class RLBenchOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        
        # Well right now we will use the joint velocity + gripper open, so 8 for panda 
        return {"actions": np.asarray(data["actions"][:, :8])}