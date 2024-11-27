from pdf2image import convert_from_path
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from torch.utils.data import DataLoader
import torch
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
    def process_queries(self, queries):
        dataloader = DataLoader(
            dataset=ListDataset[str](queries),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: self.processor.process_queries(x),
        )

        qs: List[torch.Tensor] = []

        for batch_query in dataloader:

            with torch.no_grad():
                batch_query = {k: v.to(self.model.device) for k, v in batch_query.items()}
                embeddings_query = self.model(**batch_query)
            
            qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))
        
        return qs