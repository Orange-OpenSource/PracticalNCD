from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
import wandb
import math

from src.utils import *


def smotenc_transform_batch_2(batch, cat_columns_indexes, data_queue, device, k_neighbors=5, dist='cosine', batch_size=100):
    """
    Faster but harder to understand vectorized transformation for mixed numerical and categorical data.
    See 'smotenc_transform_batch' to really understand the logic.
    Inspired from SMOTE-NC.
    :param batch: torch.Tensor : The batch data to transform.
    :param cat_columns_indexes: Array-like object of the indexes of the categorical columns.
    Only useful when transform_method='new_2'.
    :param data_queue: The unlabelled data stored in the unlab_memory_module object.
    :param device: torch.device : The device.
    :param k_neighbors: int : The number of neighbors to consider during the transformation.
    :param dist: The distance metric to use. Choices : ['cosine', 'euclidean'].
    :param batch_size: int : During computation, the batch is cut in blocs of size batch_size. If you have memory errors, reduce it.
    :return: torch.Tensor : The transformed data.
    """
    full_data = torch.cat([batch, data_queue])

    full_similarities_matrix = torch.tensor([], device=device, dtype=torch.float32)

    n_batchs = math.ceil((full_data.shape[0]) / batch_size)
    batch_start_index, batch_end_index = 0, min(batch_size, len(full_data))
    for batch_index in range(n_batchs):
        if dist == 'cosine':
            similarities = pairwise_cosine_similarity(batch, full_data[batch_start_index:batch_end_index])
        elif dist == 'euclidean':
            # ToDo (below is the non-vectorized code)
            # similarities = torch.cdist(batch[i].view(1, -1), full_data)
            # similarities[i] += torch.inf  # This way, itself wont be in the most similar instances
            # topk_similar_indexes = similarities.topk(k=k_neighbors, largest=False).indices
            pass

        full_similarities_matrix = torch.cat([full_similarities_matrix, similarities], dim=1)

        batch_start_index += batch_size
        batch_end_index = min((batch_end_index + batch_size), full_data.shape[0])

    full_similarities_matrix -= torch.eye(len(batch), len(full_data), device=device)  # This way, itself wont be in the most similar instances

    batch_topk_similar_indexes = full_similarities_matrix.topk(k=k_neighbors, dim=1).indices

    # Select a random point between the k closest
    batch_closest_point_index = torch.gather(batch_topk_similar_indexes, 1, torch.randint(low=0, high=k_neighbors, size=(len(batch),), device=device).view(-1, 1))
    batch_closest_point_index = batch_closest_point_index.flatten()

    batch_closest_point = full_data[batch_closest_point_index]

    batch_diff_vect = (batch_closest_point - batch) * torch.rand(len(batch), device=device).view(-1, 1)

    augmented_batch = batch + batch_diff_vect  # At this point, the categorical values are wrong, next line fixes that

    if len(cat_columns_indexes) > 0:
        augmented_batch[:, cat_columns_indexes] = full_data[:, cat_columns_indexes.flatten()][batch_topk_similar_indexes].mode(1).values

    return augmented_batch


class NCLMemoryModule:
    """
    A simple object to store the *M* most recent training instances from the previous batches.
    In short, this is a FIFO queue.
    This is used during the transformation, where this queue is used to have a larger pool of data from which we
    can pick the closest instance and create more realistic transformations.
    """
    def __init__(self, device, M=2000, labeled_memory=False):
        self.labeled_memory = labeled_memory
        self.current_update_idx = 0
        self.device = device
        self.queue_size = M

        self.data_memory = torch.Tensor([]).to(device)  # => The memory of the encoded data
        self.original_data_memory = torch.Tensor([]).to(device)  # => The memory of the original data
        if labeled_memory is True:
            self.labels_memory = torch.tensor([], dtype=torch.int64, device=device)

    def memory_step(self, input_data, input_original_data, input_labels=None):
        batch_size = input_data.shape[0]
        # If the memory queue isn't full yet, concatenate the batch to complete it
        if len(self.data_memory) < self.queue_size:
            len_to_cat = min(self.queue_size - len(self.data_memory), batch_size)

            self.data_memory = torch.cat((self.data_memory, input_data[:len_to_cat]))
            self.original_data_memory = torch.cat((self.original_data_memory, input_original_data[:len_to_cat]))
            if input_labels is not None:
                self.labels_memory = torch.cat((self.labels_memory, input_labels[:len_to_cat]))

            # If we have leftovers after concatenation, update memory
            if len_to_cat < batch_size:
                self.update_queue(batch_size - len_to_cat, input_data[len_to_cat:], input_original_data[len_to_cat:],
                                  input_labels[len_to_cat:] if input_labels is not None else None)
        else:
            # If the memory was full to begin with, update memory
            self.update_queue(batch_size, input_data, input_original_data, input_labels)

    def update_queue(self, batch_size, new_data, new_original_data, new_labels=None):
        indexes_to_update = torch.arange(batch_size).to(self.device)
        indexes_to_update += self.current_update_idx
        indexes_to_update = torch.fmod(indexes_to_update, self.queue_size)  # Previously incorrect : torch.fmod(indexes_to_update, *batch_size*) !!!

        self.data_memory.index_copy_(0, indexes_to_update, new_data)
        self.original_data_memory.index_copy_(0, indexes_to_update, new_original_data)
        if new_labels is not None:
            self.labels_memory.index_copy_(0, indexes_to_update, new_labels)

        self.current_update_idx = (self.current_update_idx + batch_size) % self.queue_size


class TabularNCDModel(nn.Module):
    def __init__(self, model_config):
        """
        :param encoder_layers_sizes: Size from the input to the output, not only the hidden layers.
        :param joint_learning_layers_sizes:  Only the hidden layers, as the input and output depend on the encoder and input vector size. This corresponds to the classification and clustering heads.
        :param n_known_classes: The number of known classes, or the number of output neurons of the classification head.
        :param n_unknown_classes:  The number of unknown classes, or the number of output neurons of the clustering head.
        :param activation_fct: The activation function that is *between* the first and last layers of each network. Choices : ['relu', 'sigmoid', None].
        :param encoder_last_activation_fct: The very last layer of the encoder. Choices : ['relu', 'sigmoid', None].
        :param joint_last_activation_fct: The very last layer of the classification and clustering networks. Choices : ['relu', 'sigmoid', None].
        :param p_dropout: The probability of dropout. Use p_dropout=0 for no dropout.
        """
        super(TabularNCDModel, self).__init__()

        # ==================== Encoder ====================
        layers_dims = model_config['hidden_layers_dims'] + [model_config['latent_dim']]

        self.encoder_layers = []

        self.encoder_layers.append(nn.Linear(model_config['input_size'], layers_dims[0]))

        if len(layers_dims) > 1:
            for i in range(1, len(layers_dims)):
                if model_config['activation_fct'] is not None:
                    self.encoder_layers.append(get_activation_function(model_config['activation_fct']))
                if model_config['use_batchnorm'] is True:
                    self.encoder_layers.append(nn.BatchNorm1d(num_features=layers_dims[i - 1]))
                if model_config['p_dropout'] > 0:
                    self.encoder_layers.append(nn.Dropout(p=model_config['p_dropout']))
                self.encoder_layers.append(nn.Linear(layers_dims[i - 1], layers_dims[i]))

        self.encoder_layers = nn.Sequential(*self.encoder_layers)
        # =================================================

        # ================= Joint learning ================
        self.classification_head = nn.Linear(layers_dims[-1], model_config['n_known_classes'])
        self.clustering_head = nn.Linear(layers_dims[-1], model_config['n_unknown_classes'])
        # =================================================

    def encoder_forward(self, x):
        return self.encoder_layers(x)

    def classification_head_forward(self, encoded_x):
        return self.classification_head(encoded_x)

    def clustering_head_forward(self, encoded_x):
        return self.clustering_head(encoded_x)

    def predict_new_data(self, new_data):
        self.eval()
        with torch.no_grad():
            clustering_prediction = F.softmax(self.clustering_head_forward(self.encoder_forward(new_data)), -1).argmax(dim=1).cpu()
        self.train()

        return clustering_prediction

    def joint_training(self, config,
                       x_train, y_train,
                       x_test_known, y_test_known,
                       x_test_unknown, y_test_unknown,
                       y_train_unknown,
                       unknown_class_value, use_wandb=False, disable_tqdm=False):

        losses_dict = {
            # Losses
            'full loss': [],
            'classification loss': [], 'ce loss': [], 'cs classification loss': [],
            'clustering loss': [], 'bce loss': [], 'cs clustering loss': [],

            # Performance metrics
            'train classif ACC': [], 'test classif ACC': [],
            'train cluster ACC': [], 'test cluster ACC': [],
            'train cluster NMI': [], 'test cluster NMI': [],
            'train cluster ARI': [], 'test cluster ARI': [],
        }

        device = x_train.device

        unlab_memory_module = NCLMemoryModule(device, M=config['M'], labeled_memory=False)
        lab_memory_module = NCLMemoryModule(device, M=config['M'], labeled_memory=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=config['lr'])

        cross_entropy_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        n_batchs = math.ceil((x_train.shape[0]) / config['batch_size'])
        with tqdm(range(n_batchs * config['epochs']), disable=disable_tqdm) as t:
            for epoch in range(config['epochs']):
                train_full_losses = []
                train_classification_losses = []
                train_clustering_losses = []
                train_bce_losses = []
                train_ce_losses = []
                train_cs_classification_losses = []
                train_cs_clustering_losses = []

                t.set_description(str(epoch + 1) + "/" + str(config['epochs']))

                batch_start_index, batch_end_index = 0, config['batch_size']
                for batch_index in range(n_batchs):
                    optimizer.zero_grad()

                    # (1) ===== Get the data =====
                    batch_x_train = x_train[batch_start_index:batch_end_index]
                    batch_y_train = y_train[batch_start_index:batch_end_index]

                    mask_unlab = batch_y_train == unknown_class_value
                    mask_lab = ~mask_unlab
                    assert mask_unlab.sum() > 0, "No unlabeled data in batch"

                    # Augment/Transform the data
                    with torch.no_grad():
                        augmented_x_unlab = smotenc_transform_batch_2(batch_x_train[mask_unlab], [],
                                                                      unlab_memory_module.original_data_memory, device,
                                                                      k_neighbors=config['k_neighbors'])
                        augmented_x_lab = smotenc_transform_batch_2(batch_x_train[mask_lab], [],
                                                                    lab_memory_module.original_data_memory, device,
                                                                    k_neighbors=config['k_neighbors'])

                    encoded_x = self.encoder_forward(batch_x_train)
                    encoded_x_unlab = encoded_x[mask_unlab]

                    # (2) ===== Forward the classification data and compute the losses =====
                    y_pred_lab = self.classification_head_forward(encoded_x)

                    augmented_y_pred = torch.zeros(y_pred_lab.shape, device=device)
                    encoded_augmented_x_unlab = self.encoder_forward(augmented_x_unlab)
                    augmented_y_pred[mask_unlab] = self.classification_head_forward(encoded_augmented_x_unlab)
                    encoded_augmented_x_lab = self.encoder_forward(augmented_x_lab)
                    augmented_y_pred[mask_lab] = self.classification_head_forward(encoded_augmented_x_lab)

                    ce_loss = cross_entropy_loss(y_pred_lab, torch.tensor(batch_y_train, device=device))

                    cs_loss_classifier = mse_loss(y_pred_lab, augmented_y_pred)

                    classifier_loss = config['w1'] * ce_loss + (1 - config['w1']) * cs_loss_classifier

                    # (3) ===== Forward the clustering data and compute the losses =====
                    y_pred_unlab = self.clustering_head_forward(encoded_x_unlab)

                    encoded_augmented_x_unlab = self.encoder_forward(augmented_x_unlab)
                    augmented_y_pred_unlab = self.clustering_head_forward(encoded_augmented_x_unlab)

                    # ========== Define the pseudo labels ==========
                    computed_top_k = int((config['top_k'] / 100) * len(encoded_x_unlab))

                    # Because it is symmetric, we compute the upper corner and copy it to the lower corner
                    upper_list_1, upper_list_2 = np.triu_indices(len(encoded_x_unlab), k=1)
                    unlab_unlab_similarities = nn.CosineSimilarity()(encoded_x_unlab[upper_list_1],
                                                                     encoded_x_unlab[upper_list_2])
                    similarity_matrix = torch.zeros((len(encoded_x_unlab), len(encoded_x_unlab)), device=device)
                    similarity_matrix[upper_list_1, upper_list_2] = unlab_unlab_similarities
                    similarity_matrix += similarity_matrix.T.clone()

                    top_k_most_similar_instances_per_instance = similarity_matrix.argsort(descending=True)[:, :computed_top_k]

                    pseudo_labels_matrix = torch.zeros((len(encoded_x_unlab), len(encoded_x_unlab)), device=device)
                    pseudo_labels_matrix = pseudo_labels_matrix.scatter_(index=top_k_most_similar_instances_per_instance, dim=1, value=1)

                    # The matrix isn't symmetric, because the graph is directed
                    # So if there is one link between two points, regardless of the direction, we consider this pair to be positive
                    pseudo_labels_matrix += pseudo_labels_matrix.T.clone()
                    pseudo_labels_matrix[pseudo_labels_matrix > 1] = 1  # Some links will overlap
                    pseudo_labels = pseudo_labels_matrix[upper_list_1, upper_list_2]
                    # ==============================================

                    bce_loss = unsupervised_classification_loss(y_pred_unlab[upper_list_1], y_pred_unlab[upper_list_2], pseudo_labels)

                    cs_loss_clustering = mse_loss(y_pred_unlab, augmented_y_pred_unlab)

                    clustering_loss = config['w2'] * bce_loss + (1 - config['w2']) * cs_loss_clustering

                    full_loss = classifier_loss + clustering_loss

                    # Backward
                    full_loss.backward()
                    optimizer.step()

                    # Save losses for plotting purposes
                    train_full_losses.append(full_loss.item())
                    train_classification_losses.append(classifier_loss.item())
                    train_clustering_losses.append(clustering_loss.item())
                    train_bce_losses.append(bce_loss.item())
                    train_ce_losses.append(ce_loss.item())
                    train_cs_classification_losses.append(cs_loss_classifier.item())
                    train_cs_clustering_losses.append(cs_loss_clustering.item())

                    t.set_postfix_str("full={:05.3f}".format(np.mean(train_full_losses))
                                      + " classif={:05.3f}".format(np.mean(train_classification_losses))
                                      + " clust={:05.3f}".format(np.mean(train_clustering_losses))
                                      + " ce={:05.3f}".format(np.mean(train_ce_losses))
                                      + " bce={:05.3f}".format(np.mean(train_bce_losses))
                                      + " cs1={:05.3f}".format(np.mean(train_cs_classification_losses))
                                      + " cs2={:05.3f}".format(np.mean(train_cs_clustering_losses)))
                    t.update()

                    # Update the memory modules
                    unlab_memory_module.memory_step(encoded_x_unlab.detach().clone(), batch_x_train[mask_unlab].detach().clone())
                    lab_memory_module.memory_step(encoded_x[mask_lab].detach().clone(), batch_x_train[mask_lab].detach().clone(),
                                                  input_labels=torch.tensor(batch_y_train[mask_lab], device=device))

                    batch_start_index += config['batch_size']
                    batch_end_index = min((batch_end_index + config['batch_size']), x_train.shape[0])

                # Losses
                losses_dict['full loss'].append(np.mean(train_full_losses))
                losses_dict['classification loss'].append(np.mean(train_classification_losses))
                losses_dict['clustering loss'].append(np.mean(train_clustering_losses))
                losses_dict['bce loss'].append(np.mean(train_bce_losses))
                losses_dict['ce loss'].append(np.mean(train_ce_losses))
                losses_dict['cs classification loss'].append(np.mean(train_cs_classification_losses))
                losses_dict['cs clustering loss'].append(np.mean(train_cs_clustering_losses))

                # Performance metrics
                self.eval()
                with torch.no_grad():
                    y_train_known_pred = self.classification_head_forward(self.encoder_forward(x_train[y_train != unknown_class_value])).argmax(-1).cpu().numpy()
                    train_classif_acc = accuracy_score(y_train[y_train != unknown_class_value], y_train_known_pred)

                    y_test_known_pred = self.classification_head_forward(self.encoder_forward(x_test_known)).argmax(-1).cpu().numpy()
                    test_classif_acc = accuracy_score(y_test_known, y_test_known_pred)

                    y_train_unknown_pred = self.clustering_head_forward(self.encoder_forward(x_train[y_train == unknown_class_value])).argmax(-1).cpu().numpy()
                    train_cluster_acc = hungarian_accuracy(y_train_unknown, y_train_unknown_pred)
                    train_cluster_nmi = normalized_mutual_info_score(y_train_unknown, y_train_unknown_pred)
                    train_cluster_ari = adjusted_rand_score(y_train_unknown, y_train_unknown_pred)

                    y_test_unknown_pred = self.clustering_head_forward(self.encoder_forward(x_test_unknown)).argmax(-1).cpu().numpy()
                    test_cluster_acc = hungarian_accuracy(y_test_unknown, y_test_unknown_pred)
                    test_cluster_nmi = normalized_mutual_info_score(y_test_unknown, y_test_unknown_pred)
                    test_cluster_ari = adjusted_rand_score(y_test_unknown, y_test_unknown_pred)
                self.train()

                losses_dict['train classif ACC'].append(train_classif_acc)
                losses_dict['test classif ACC'].append(test_classif_acc)

                losses_dict['train cluster ACC'].append(train_cluster_acc)
                losses_dict['test cluster ACC'].append(test_cluster_acc)

                losses_dict['train cluster NMI'].append(train_cluster_nmi)
                losses_dict['test cluster NMI'].append(test_cluster_nmi)

                losses_dict['train cluster ARI'].append(train_cluster_ari)
                losses_dict['test cluster ARI'].append(test_cluster_ari)

                if use_wandb is True:
                    wandb.log({
                        "full loss": np.mean(train_full_losses),
                        "classification loss": np.mean(train_classification_losses),
                        "clustering loss": np.mean(train_clustering_losses),
                        "bce loss": np.mean(train_bce_losses),
                        "ce loss": np.mean(train_ce_losses),
                        "cs classification loss": np.mean(train_cs_classification_losses),
                        "cs clustering loss": np.mean(train_cs_clustering_losses),

                        "train classif ACC": train_classif_acc, "test classif ACC": test_classif_acc,
                        "train cluster ACC": train_cluster_acc, "test cluster ACC": test_cluster_acc,
                        "train cluster NMI": train_cluster_nmi, "test cluster NMI": test_cluster_nmi,
                        "train cluster ARI": train_cluster_ari, "test cluster ARI": test_cluster_ari,
                    })

        return losses_dict
