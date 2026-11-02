# Few-Shot Component Recognition with Prototypical Networks

A MATLAB implementation of **Prototypical Networks** for few-shot image classification of manufacturing components. The project investigates whether a CNN embedding model trained on a set of base component classes can recognize **previously unseen component classes** from only a few labeled examples.

The implementation uses pretrained **GoogLeNet** or **ResNet-18** as a feature extractor, learns a **128-dimensional embedding** through episodic training, and evaluates classification on a separate set of novel component images.

## Research objective

Recognizing new component geometries can be challenging when collecting and labeling a large training dataset is impractical. Few-shot learning addresses this by learning a representation in which a small labeled *support set* can define a new classification task.

This project explores that approach for manufacturing component images. The research dataset includes base component categories involving plates and pipes, with novel categories involving square-pipe configurations. The exact classes used in a run come from the image-folder names; they are not hardcoded in the network.

## Method

The model is trained using **N-way, K-shot episodes**:

1. Sample `N` classes and `K` labeled support images per class, along with `Q` query images per class.
2. Pass the support and query images through the CNN to obtain embedding vectors.
3. Average the support embeddings from each class to produce a **class prototype**.
4. Compute distances between query embeddings and the prototypes.
5. During training, optimize the network using a distance-based classification loss. During testing, assign each query image to its closest prototype.

The classifier is formed from the support images at evaluation time; a fixed output class head is not trained for the novel categories.

### Experimental configuration

| Setting | Training | Novel-class evaluation |
| --- | --- | --- |
| Classes per episode (`N`) | 7 | 5 |
| Support images per class (`K`) | 5 | 5 |
| Query images per class (`Q`) | 10 | 10 |
| Input resolution | 224 × 224 | 224 × 224 |
| Embedding dimension | 128 | 128 |
| Number of episodes | 50 epochs × 10 episodes | 500 episodes |
| Data augmentation | Random X/Y reflections and rotation from −90° to 90° | None |
| Class assignment | Squared Euclidean-distance logits and cross-entropy loss | Nearest prototype by Euclidean distance |

Training uses the **Adam optimizer** with a model-specific learning-rate schedule. The available backbone choices are:

| Input at training prompt | Backbone | Initial learning rate | Freeze setting |
| --- | --- | --- | --- |
| `g` | GoogLeNet | `0.00005` | No layer freezing requested |
| `gf` | GoogLeNet | `0.0001` | Attempt to freeze first 50% of layers |
| `r` | ResNet-18 | `0.0001` | No layer freezing requested |
| `rf` | ResNet-18 | `0.0002` | Attempt to freeze first 50% of layers |

The frozen options use the project's `freezeFirstPercent.m` implementation. The ResNet variants also include dropout (`0.4`) before the embedding layer.

## Requirements

- MATLAB with **Deep Learning Toolbox**
- **Image Processing Toolbox** (`imresize`)
- **Statistics and Machine Learning Toolbox** (`datasample`, `pdist2`)
- **Parallel Computing Toolbox** and a compatible GPU for the current `gpuArray` implementation
- Pretrained **GoogLeNet** and/or **ResNet-18** model support packages, depending on the selected architecture
- A MATLAB release supporting `dlnetwork`, `dlfeval`, `adamupdate`, and `trainingProgressMonitor`

The exact MATLAB version and GPU environment used for the original experiments are not documented here. The supplied scripts have not been independently validated across MATLAB releases.

## Dataset organization

Images are read with `imageDatastore`, using **subfolder names as class labels**. Organize the dataset as follows:

```text
MATLAB-Few-Shot-Learning-minimal/
├── base/
│   ├── class_1/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── class_2/
│       └── ...
├── novel/
│   ├── novel_class_1/
│   │   └── ...
│   └── novel_class_2/
│       └── ...
├── FSLtrain.m
├── FSLtest.m
└── ...
```

The training set requires at least **7 classes** and the novel evaluation set at least **5 classes**, each with at least **15 images** to sample one episode without replacement (5 support + 10 query). Use distinct base and novel class categories to assess generalization to unseen classes. RGB images are the most compatible with the current training preprocessing.

The image datasets and saved network files are **not included** in this repository.

## Running the experiments

1. Clone or download the repository and set the MATLAB **Current Folder** to the repository directory. All `.m` files are kept together; no packages or path setup are required.
2. Add your datasets as `base/` and `novel/`, using the folder structure above.
3. Run the training script:

   ```matlab
   FSLtrain
   ```

4. At the prompt, enter `g`, `gf`, `r`, or `rf`. Training progress is displayed in the MATLAB Training Progress Monitor. When prompted, enter `y` to save the trained network to the Current Folder.
5. Run novel-class evaluation:

   ```matlab
   FSLtest
   ```

6. Enter the matching architecture selection. The script reports **mean episode accuracy** and **standard deviation** across the sampled evaluation episodes.

Saved network filenames are `TrainedFSLNet_googlenet.mat`, `TrainedFSLNet_googlenetf.mat`, `TrainedFSLNet_resnet18.mat`, and `TrainedFSLNet_resnet18f.mat` for `g`, `gf`, `r`, and `rf`, respectively.

## Code organization

| File | Responsibility |
| --- | --- |
| [`FSLtrain.m`](FSLtrain.m) | Construct the embedding network, load the base dataset, and run episodic training |
| [`FSLtest.m`](FSLtest.m) | Load a saved network and evaluate novel-class episodes |
| [`episodemakerTrain.m`](episodemakerTrain.m) | Sample and augment training support/query sets |
| [`episodemakerTest.m`](episodemakerTest.m) | Sample support/query indices for evaluation |
| [`dlarrayMakerTrain.m`](dlarrayMakerTrain.m) | Assemble training data into GPU `dlarray` batches |
| [`dlarrayMakerTest.m`](dlarrayMakerTest.m) | Assemble evaluation data into GPU `dlarray` batches |
| [`gradientsandloss.m`](gradientsandloss.m) | Compute prototypes, loss, episode accuracy, and gradients |
| [`freezeFirstPercent.m`](freezeFirstPercent.m) | Set learning-rate factors on selected pretrained layers |

The code uses standard MATLAB functions in separate `.m` files, rather than a package namespace.

## Evaluation notes and limitations

- **Review the evaluation image-loading loop before reporting new experimental results.** The current `FSLtest.m` allocates an array for the full novel dataset but loads only the first 100 images. If the dataset size differs, this can cause an error or incorrect measurements. This issue is retained in the current code and has not been fixed.
- Evaluation episodes are randomly sampled and no random seed is set, so results may vary between runs.
- The implementation depends on GPU execution; CPU fallback is not currently provided.
- An invalid network choice in `FSLtrain.m` enters an unsafe legacy branch that repeatedly allocates a very large array. **Use only `g`, `gf`, `r`, or `rf`.**
- A separate transfer-learning baseline and visualization scripts are outside the scope of this repository.

## Research context

This code is part of an investigation into **representation learning and generalization across manufacturing component geometries**. The focus of this repository is the Prototypical Network approach: learning an embedding from base-class episodes and forming new class prototypes from limited labeled support images at test time.
