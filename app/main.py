import base64
import logging
from typing import Dict

from core.database import DatabaseManager
from core.llm import query_llm
from core.stub import Stub
from core.utils import save_3d_model, save_image
from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()

# Initialise the database manager
db_manager = DatabaseManager()


############################################################
# Config callback function
############################################################
def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Stores user-specific configuration data.

    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application (not used in this implementation).
    """
    for uid, conf in configuration.items():
        logging.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf


############################################################
# Execution callback function
############################################################
def execute(model: AppModel) -> None:
    """
    Main execution entry point for handling a model pass.

    Args:
        model (AppModel): The model object containing request and response structures.
    """

    # Retrieve input
    request: InputClass = model.request

    # Retrieve user config
    user_config: ConfigClass = configurations.get("super-user", None)
    logging.info(f"{configurations}")

    similar_generations = db_manager.find_similar_generations(request.prompt)
    logging.info(f"Similar generations found: {similar_generations}")

    # Expand prompt with LLM
    expanded_prompt = query_llm(request.prompt, similar_generations)
    logging.info(f"Expanded prompt: {expanded_prompt}")

    # Initialize the Stub with app IDs
    app_ids = user_config.app_ids if user_config else []
    stub = Stub(app_ids)
    text_to_image_res = stub.call(app_ids[0], data={"prompt": expanded_prompt})

    # Save image for 3D conversion
    image_path = save_image(app_ids[0], text_to_image_res["result"])
    logging.info("Generated image saved to disk")

    # Convert image to 3D model
    with open(image_path, "rb") as img_file:
        img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

    # Call Image-to-3D conversion app
    image_to_3d_res = stub.call(app_ids[1], data={"input_image": img_base64})
    save_3d_model(app_ids[1], image_to_3d_res["generated_object"])
    logging.info("3D model saved to disk")

    # Persist the generation in the database
    db_manager.save_generation(request.prompt, expanded_prompt)
    logging.info("Generation saved to database")

    # Prepare response
    response: OutputClass = model.response
    response.message = image_to_3d_res
