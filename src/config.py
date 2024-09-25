device = 'cuda'

# create loader
list_labels_cat = ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'monocyte', 'neutrophil']
reference_path = '../references'
source = ['barcelona', 'rabin','bccd']
size = [1000 for i in range(len(source))]
balanced = False
transform = True
fine_tune = True
full_evaluation = False



# Load model
model_path = '../models/Mixed_sources_trained/barcelona_matek_2018/efficientnet_1_transform3_wd_01_reduced'

# Train or fine tune model
batchsize = 5
epoch = 50
lr = 7e-5

# Visualisation of the results
create_confusion = False  # Set to True to generate the confusion matrix, along with accuracy, recall, and precision per class.
plot_confusion = False     # Set to True to visualize the confusion matrix.
scatter_embeddings = False  # Set to True to create a scatter plot of the model's embeddings.
show_group_from_region = False  # Set to True to display cells corresponding to a specific region in the scatter plot of embeddings.
show_flattened_images = False  # Set to True to create a scatter plot of flattened images.
bootstrap = True            # Set to True to perform bootstrap analysis on predictions.
show_proba_per_image = False  # Set to True to display a histogram of predicted probabilities for each image.
show_auc_per_class = True   # Set to True to display the Precision-Recall curve for each class in the dataset.
show_auc_all = False        # Set to True to display all Precision-Recall curves for all datasets.
method = 'UMAP'             # Method used for dimensionality reduction (e.g., UMAP or t-SNE).
