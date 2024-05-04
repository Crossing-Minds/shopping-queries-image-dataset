# Shopping Queries Image Dataset (SQID 🦑): An Image-Enriched ESCI Dataset for Exploring Multimodal Learning in Product Search


## Introduction
The Shopping Queries Image Dataset (SQID) is a dataset that includes image information for over 190,000 products. This dataset is an augmented version of the [Amazon Shopping Queries Dataset](https://github.com/amazon-science/esci-data), which includes a large number of product search queries from real Amazon users, along with a list of up to 40 potentially relevant results and judgments of how relevant they are to the search query.

The image-enriched SQID dataset can be used to support research on improving product search by leveraging image information. Researchers can use this dataset to train multimodal machine learning models that can take into account both textual and visual information when ranking products for a given search query.

## Dataset
This dataset extends the Shopping Queries Dataset (SQD) by including image information and visual embeddings for each product, as well as text embeddings for the associated queries which can be used for baseline product ranking benchmarking. 

### Product Sampling
We limited this dataset to the subset of the SQD where `small_version` is 1 (the reduced version of the dataset for Task 1), `split` is 'test' (test set of the dataset), and `product_locale` is 'us'. 
Hence, this dataset includes 164,900 `product_id`s.

As supplementary data, we also provide data related to the other products appearing in at least 2 query judgements in the data of Task 1 with `product_locale` as 'us', amounting to 27,139 products, to further increase the coverage of the data for additional applications that go beyond the ESCI benchmark.

## Image URL Scraping

We scraped 156,545 (~95% of the 164,900 `product_id`'s) `image_url`s from the Amazon website. Products lacking `image_url`s either failed to fetch a valid product page (usually if Amazon no longer sells the product) or displayed a default "No image available" image.

Note: 442 product `image_url`s are a default digital video image, `'https://m.media-amazon.com/images/G/01/digital/video/web/Default_Background_Art_LTR._SX1080_FMjpg_.jpg'`, implying no product-specific image exists.

The SQID dataset also includes a supplementary file covering 27,139 more products.

### Image Embeddings

We extracted image embeddings for each of the images using the [OpenAI CLIP model from HuggingFace](https://huggingface.co/openai/clip-vit-large-patch14), specifically clip-vit-large-patch14, with all default settings. 

### Text Embeddings

For each query and each product in the SQD Test Set, we extracted text embeddings using the same CLIP model and based on the query text and product title. These can be useful to benchmark a baseline product search method where both text and images share the same embedding space.

## Files
The `sqid` directory contains 4 files:
- `product_image_urls.csv`
  - This file contains the image URLs for all product_id's in the dataset, forming the test set of the ESCI dataset
- `product_features.parquet`
  - This file contains the CLIP embedding features for product_id's in the dataset
- `query_features.parquet`
  - This file contains the CLIP text embedding features for queries in the dataset
- `supp_product_image_urls.csv`
  - This file contains supplementary data as image URLs for an additional set of products not included in the test set and increasing the coverage of the data

## Code snippets to get CLIP features

SQID includes embeddings extracted using [OpenAI CLIP model from HuggingFace](https://huggingface.co/openai/clip-vit-large-patch14) (clip-vit-large-patch14). We provide below code snippets in Python to extract such embeddings, using either the model from HuggingFace or using [Replicate](https://replicate.com/).

### Using CLIP model from HuggingFace

```
from PIL import Image
import requests
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

image = Image.open(requests.get('https://m.media-amazon.com/images/I/71fv4Dv5RaL._AC_SY879_.jpg', stream=True).raw)
inputs = processor(images=[image], return_tensors="pt", padding=True)
image_embds = model.get_image_features(pixel_values=inputs["pixel_values"])
```

### Using Replicate
```
import replicate

client = replicate.Client(api_token=REPLICATE_API_KEY)
output = client.run(
    "andreasjansson/clip-features:71addf5a5e7c400e091f33ef8ae1c40d72a25966897d05ebe36a7edb06a86a2c",
    input={
        "inputs": 'https://m.media-amazon.com/images/I/71fv4Dv5RaL._AC_SY879_.jpg'
    }
)
```

## Citation
To use this dataset, please cite the following paper:
<pre>
Shopping Queries Image Dataset (SQID): An Image-Enriched ESCI Dataset for Exploring Multimodal Learning in Product Search, M. Al Ghossein, C.W. Chen, J. Tang
</pre>


## License
This dataset is released under the MIT License

## Acknowledgments
SQID was developed at [Crossing Minds](https://www.crossingminds.com) by:
- [Marie Al Ghossein](https://www.linkedin.com/in/mariealghossein/)
- [Ching-Wei Chen](https://www.linkedin.com/in/cweichen)
- [Jason Tang](https://www.linkedin.com/in/jasonjytang/)

This dataset would not have been possible without the amazing [Shopping Queries Dataset by Amazon](https://github.com/amazon-science/esci-data).
