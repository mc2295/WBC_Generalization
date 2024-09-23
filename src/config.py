device = 'cuda'

# create loader
list_labels_cat = ['basophil', 'eosinophil', 'erythroblast', 'lymphocyte', 'monocyte', 'neutrophil']
reference_path = '../references'
source = ['matek', 'rabin','bccd', 'tianjin_reviewed']
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
show_confusion = False
plot_confusion = False
scatter_embeddings = False
show_group_from_region = False
show_flattened_images = False
bootstrap = True
show_proba_per_image = False
show_auc_per_class = True
show_auc_all = False
method = 'UMAP'