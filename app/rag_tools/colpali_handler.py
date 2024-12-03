from pdf2image import convert_from_path
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from typing import List, cast

class ColpaliHandler:
    '''
    Custom Embedder class meant for managing our colpali functions

    '''

    def __init__(self, device, col_pali): 

        self.device = get_torch_device(device)
        self.model_name = "vidore/colpali-v1.2"

        self.model = ColPali.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        ).eval()

        self.processor = cast(ColPaliProcessor, ColPaliProcessor.from_pretrained(self.model_name))



    # Accepts a list of images (rasterized Pdf pages) returns embeddings
    def process_images(self, images):
        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_images(x),
        )

        ds: List[torch.Tensor] = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(self.model.device) for k, v in batch_doc.items()}
                embeddings_doc = self.model(**batch_doc)

            ds.extend(list(torch.unbind(embeddings_doc.to("cpu"))))
        
        return ds
    

    # Accepts a list of queries to be embedded to tensors returns embeddings
    def process_query(self, query):
        # Process the query using the processor (if necessary)
        processed_query = self.processor.process_queries([query])  # wrap query in a list
        
        # Convert to appropriate device
        processed_query = {k: v.to(self.model.device) for k, v in processed_query.items()}
        
        # Forward pass through the model to get embeddings
        with torch.no_grad():
            embeddings_query = self.model(**processed_query)
        
        # Extract the embeddings (assuming they are in the form of a tensor)
        q = torch.unbind(embeddings_query.to("cpu"))
        
        return q