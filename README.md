# Green Roof Segmentation in Milan

A deep-learning project at the intersection of GIS and computer vision, developing semantic-segmentation models to identify roofs suitable for green-roof conversion from aerial imagery.

> **Academic context:** Developed in 2025 by Lorenzo De Luca and Edoardo Guida as a two-person university project.

**Technologies:** Python, TensorFlow/Keras, OpenCV, NumPy, GeoJSON, Playwright, Azure Maps

---

## Table of Contents

- [Dataset](#dataset)
- [Deep Learning Models](#deep-learning-models)
- [Future Developments](#future-developments)
- [Requirements](#requirements)
- [Usage](#usage)

## Dataset

### 1. Ground Truth Data

The starting point was to establish ground-truth labels for roofs considered potentially suitable for green-roof conversion. The [Municipality of Milan's GeoJSON dataset](https://dati.comune.milano.it/dataset/ds1446_tetti-verdi-potenziali) provides the corresponding polygon coordinates, which we processed and cleaned for the analysis.

### 2. Choice of Underlying Aerial Imagery

We used the GeoJSON polygons as masks over aerial images of Milan to generate the model-training dataset. This required high-resolution imagery that aligned closely with the municipal coordinates. Finding a suitable source was challenging because publicly accessible maps differed in both resolution and spatial alignment.

We tested the overlay against imagery from Google Maps, Microsoft Azure Maps, and OpenStreetMap. Azure Maps provided the best combination of resolution and alignment with the GeoJSON polygons.

| Azure Map               | Google Map                 |
|-------------------------|----------------------------|
| ![Azure Map](images/azure.jpg) | ![Google Map](images/google.jpg) |

### 3. Map Preparation

We created a small web application in `web/index.html` for panning across the aerial map and toggling the GeoJSON mask. A browser-based viewer was the most direct way to work with the Azure Maps Web SDK and also provided the surface used for automated data collection.

Run `npm run serve`, then open `http://localhost:3000/web/` to view it locally.

### 4. Dataset Collection

We used Playwright to automate paired map capture. The script in `scripts/capture_map.js` visits deterministically sampled coordinates, captures the aerial image, enables the polygon overlay, and captures the matching label view at the same position. It stores the resulting pairs under `data/generated/` and records their coordinates and capture settings in a JSONL manifest.

During the original experiment, the collector captured approximately five screenshots per second with a reliable internet connection.

Some example images are shown below:

| Aerial image | Polygon overlay |
|-----------------|--------------|
| ![Image 1](images/45.46002_9.18572.jpg) | ![Image 2](images/45.46002_9.18572-1.jpg) |

### 5. Dataset Preprocessing

The final data-preparation step converts each rendered overlay into a binary mask. Pixels belonging to a potentially suitable roof are encoded as 1, while all other pixels are encoded as 0.

The original collection contained 5,000 paired captures—10,000 raw 600 × 600 PNG files—which occupied approximately 5 GB.

To reduce storage requirements and improve training throughput, we converted the images and masks into NumPy arrays and stored them in a compressed NPZ file.

The preprocessing logic extracts the magenta polygon overlay as the positive class:

```python
def preprocess_image_and_mask(image_path, mask_path, target_size=(600, 600)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    magenta = np.array([255, 0, 255])
    binary_mask = np.all(mask_rgb == magenta, axis=-1).astype(np.uint8)
    binary_mask = cv2.resize(binary_mask, target_size, interpolation=cv2.INTER_NEAREST)
    binary_mask = np.expand_dims(binary_mask, axis=-1)  # Add channel dimension

    return image, binary_mask
```

| Aerial image | Binary mask |
|-----------------|--------------|
| ![Image 1](images/Sat1.jpeg) | ![Image 2](images/Mask1.jpeg) |

## Deep Learning Models

### 1. Approach and Methodology

With the dataset prepared, we used **transfer learning** to develop the segmentation models. Given the resolution of the source captures, both approaches accept 600 × 600 × 3 inputs.

Two models were chosen for experimentation:
- **EfficientNetB7**: A highly optimized CNN known for its strong feature extraction capabilities.
- **ResNet50 + ASPP**: A segmentation encoder-decoder inspired by DeepLabV3+, combining a ResNet50 backbone with Atrous Spatial Pyramid Pooling.

We experimented with different hyperparameters and two loss functions: Binary Cross-Entropy (BCE) and Focal Loss.

- Binary Cross-Entropy (BCE): This is the standard loss function for binary classification problems. It measures the difference between predicted probabilities and actual labels, penalizing incorrect predictions linearly.

- Focal Loss: A variation of BCE that introduces a focusing parameter (gamma) to reduce the relative importance of well-classified examples, thereby helping to handle class imbalance in segmentation tasks.

> **Evaluation note:** The metrics below are retained from the original university experiment as historical exploratory results. The cleaned pipeline now uses roof-positive masks and geographically grouped data splits, so the recorded values should not be treated as directly reproducible production benchmarks.

### 2. Model Architectures

#### 2.1. EfficientNetB7

We used EfficientNetB7 as a feature extractor, removing its top classification layer and keeping the pre-trained weights frozen to prevent overfitting. The extracted features were then passed through a custom decoder built with Conv2DTranspose layers for upsampling.

##### **Model Implementation**
```python
def build_model(input_shape=(600, 600, 3)):
    base_model = EfficientNetB7(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze pre-trained weights

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)

    # Decoder with Conv2DTranspose for upsampling
    x = layers.Conv2DTranspose(256, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(16, (3, 3), strides=2, padding='same', activation='relu')(x)

    x = layers.Cropping2D(((4, 4), (4, 4)))(x)

    # Output layer with sigmoid activation for binary segmentation
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    return model
```

##### **Training, Compilation and Evaluation**
We used a dataset of 5,000 non-normalized images for training. After numerous experiments with different configurations, this was found to be the best setup.

```python
optimizer = 'adam'
loss = keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False)
metrics = ['accuracy', iou_metric]
batch_size = 20
epochs = 100
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```
| Metrics | Value |
|:---------:|:---------:|
|  Accuracy   |   0.9129       |
|  Precision  |   0.9614       |
|  Recall     |   0.9400       |
|  IoU        |   0.6777       |

| Aerial image | Ground truth | Model output |
|---------------------------------|---------------------------------|---------------------------------|
| ![Image 1](images/Sat1.jpeg) | ![Image 2](images/Mask1.jpeg) | ![Image 3](images/Eff1.jpeg) |
| ![Image 4](images/Sat2.jpeg) | ![Image 5](images/Mask2.jpeg) | ![Image 6](images/Eff2.jpeg) |

#### 2.2. ResNet50 + ASPP

The second approach was inspired by DeepLabV3+. It uses ResNet50 as its feature-extraction backbone and an ASPP-style module to capture spatial context at multiple scales.

Atrous Spatial Pyramid Pooling applies dilated convolutions at different dilation rates to balance fine-grained details with broader spatial context. This is useful when segmenting roof regions of different sizes in high-resolution aerial imagery.

##### **Model Implementation**
```python
def build_model(input_shape=(600, 600, 3), num_classes=1):
    base_model = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze ResNet50 weights

    features = base_model.get_layer("conv4_block6_out").output

    # Parallel ASPP-style branches
    branches = [layers.Conv2D(256, 1, padding="same", activation="relu")(features)]
    for rate in (6, 12, 18):
        branch = layers.Conv2D(
            256, 3, dilation_rate=rate, padding="same", use_bias=False
        )(features)
        branch = layers.BatchNormalization()(branch)
        branches.append(layers.ReLU()(branch))

    x = layers.Concatenate()(branches)
    x = layers.Conv2D(256, 1, padding="same", activation="relu")(x)

    # Decoder (Upsampling)
    x = layers.Conv2DTranspose(256, (3, 3), strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, (3, 3), strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, (3, 3), strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, (3, 3), strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Cropping2D(((4, 4), (4, 4)))(x)

    # Final output layer with sigmoid activation
    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(x)

    model = keras.Model(inputs=base_model.input, outputs=outputs)
    return model
```

#### **Training and Compilation**

We used 3,000 normalized images for this experiment—fewer than for EfficientNetB7 because of the limited resources of the development environment.

```python
optimizer = 'adam'
loss = 'binary_crossentropy'
metrics = ['accuracy', iou_metric]
batch_size = 8
epochs = 50
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

```
| Metrics | Value |
|:---------:|:---------:|
|  Accuracy   |   0.9348       |
|  Precision  |   0.9561       |
|  Recall     |   0.9706       |
|  IoU        |   0.7378       |

| Aerial image | Ground truth | Model output |
|---------------------------------|---------------------------------|---------------------------------|
| ![Image 1](images/Sat1.jpeg) | ![Image 2](images/Mask1.jpeg) | ![Image 3](images/Res1.jpeg) |
| ![Image 4](images/Sat2.jpeg) | ![Image 5](images/Mask2.jpeg) | ![Image 6](images/Res2.jpeg) |

### 3. Ensemble Model

To improve the IoU obtained by the individual models, we averaged their probability maps before thresholding the final prediction. This experimental ensemble combined EfficientNetB7's feature extraction with the ResNet50 model's multi-scale spatial processing and produced cleaner segmentation boundaries in the original evaluation.

| Metrics | Value |
|:---------:|:---------:|
|  Accuracy   |   0.9500       |
|  Precision  |   0.9759       |
|  Recall     |   0.9668      |
|  IoU        |   0.8251      |

| Aerial image | Ground truth | Model output |
|---------------------------------|---------------------------------|---------------------------------|
| ![Image 1](images/Sat1.jpeg) | ![Image 2](images/Mask1.jpeg) | ![Image 3](images/Ens1.jpeg) |
| ![Image 4](images/Sat2.jpeg) | ![Image 5](images/Mask2.jpeg) | ![Image 6](images/Ens2.jpeg) |

## Future Developments

The project demonstrates the feasibility of using deep learning to segment potentially suitable green-roof areas from high-resolution aerial imagery. Several extensions could improve its generalization and practical usability.

### Testing on Other Cities

The models were trained and tested using data from Milan, where the aerial images and ground-truth masks were closely aligned. Evaluating them in other cities would provide a stronger measure of geographic generalization.

- If the model performs well, it confirms its robustness in different urban landscapes.
- If the model struggles, it may indicate a need for fine-tuning or retraining using data from diverse locations.
Training with data from multiple cities could make the models more adaptable and reduce bias toward a single urban environment.

### Improving Mask Post-Processing

The current output consists of pixel-wise binary masks, while spatial-planning and GIS workflows generally require georeferenced vector polygons.

A possible future improvement is to apply post-processing techniques to refine the segmentation results:

- **Morphological cleanup:** Remove small artifacts and smooth irregular mask boundaries.
- **Contour extraction and polygon approximation:** Convert raster masks into vector polygons.
- **Georeferencing and topology validation:** Produce geometries suitable for downstream GIS analysis.

## Requirements

- Python 3.11–3.13
- Node.js 18+
- An Azure Maps account for regenerating aerial-image pairs

## Usage

### 1. Install dependencies

```sh
uv sync --extra dev --extra training
npm ci
npx playwright install chromium
```

### 2. Prepare the municipal GeoJSON

Download the [potential green roofs dataset](https://dati.comune.milano.it/dataset/ds1446_tetti-verdi-potenziali), then normalize it:

```sh
uv run greenroof-clean-geojson data/raw/potential_green_roofs.geojson \
  data/generated/cleaned_potential_green_roofs.geojson
```

### 3. Configure and serve the map

```sh
cp web/config.example.js web/config.js
# Add a restricted Azure Maps development key to web/config.js.
npm run serve
```

The local `web/config.js` file is ignored by Git so credentials are never committed.

### 4. Capture paired imagery

In another terminal:

```sh
npm run capture -- --count 5000 --seed 42
```

### 5. Build the compressed dataset

```sh
uv run greenroof-preprocess \
  data/generated/images \
  data/generated/overlays \
  data/preprocessed/dataset.npz
```

### 6. Train a model

```sh
uv run python scripts/train.py \
  --dataset data/preprocessed/dataset.npz \
  --model deeplab \
  --output models/deeplab.keras
```

The cleaned training entry point uses geographically grouped splits to reduce leakage between overlapping image tiles.
