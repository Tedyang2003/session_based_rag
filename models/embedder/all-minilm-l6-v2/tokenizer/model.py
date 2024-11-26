import os
from typing import Dict, List

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer, PreTrainedTokenizer, TensorType
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class TritonPythonModel:
    tokenizer: PreTrainedTokenizer

    def initialize(self, args: Dict[str, str]) -> None:
        """
        Initialize the tokenization process
        :param args: arguments from Triton config file
        """
        path: str = os.path.join(args["model_repository"], args["model_version"])
        self.tokenizer = AutoTokenizer.from_pretrained(path)

    def execute(self, requests) -> "List[List[pb_utils.Tensor]]":
        """
        Parse and tokenize each request
        :param requests: 1 or more requests received by Triton server.
        :return: text as input tensors
        """
        responses = []
        # for loop for batch requests (disabled in our case)
        for request in requests:
            # binary data typed back to string
            query = [
                t.decode("UTF-8")
                for t in pb_utils.get_input_tensor_by_name(request, "TEXT") 
                .as_numpy()
                .tolist()
            ]

            # Tokenize String into tokenizer items
            tokens: Dict[str, np.ndarray] = self.tokenizer(
                text=query, return_tensors=TensorType.NUMPY
            )
            # tensorrt uses int32 as input type, ort uses int64
            # For each item in the tokenizer take out the components as a dictionary
            tokens = {k: v.astype(np.int64) for k, v in tokens.items()}

            # prepare payload for the tokenizer on what the model transformer needs: input_id & attention_mask
            outputs = list()
            for input_name in self.tokenizer.model_input_names:
                logging.info(input_name)
                logging.info(tokens[input_name])
                tensor_input = pb_utils.Tensor(input_name, tokens[input_name])
                outputs.append(tensor_input)

            # Append the response as an inference response
            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)
        
        logging.info(responses)
        return responses