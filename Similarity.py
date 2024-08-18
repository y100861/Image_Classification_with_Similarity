!pip install faiss-gpu
import faiss

# Gallery 
gallery_emb, gallery_label = [], []

with torch.no_grad():
    for input, label in trainloader:
        input, label = input.to(device), label.to(device)
        emb = best_model.feat_extractor(input)
        emb = torch.flatten(emb, 1)
        print(emb.shape)

        for i in range(len(label)):
            gallery_emb.append(emb[i].cpu().numpy())
            gallery_label.append(label[i].cpu().numpy())

gallery_emb = np.vstack(gallery_emb)
gallery_label = np.array(gallery_label)


# Query
with torch.no_grad():
    q_input, q_label = val_data[0]
    q_input = q_input.unsqueeze(0).to(device) # C, H, W -> N, C, H, W
    query = best_model.feat_extractor(q_input)
    query = torch.flatten(query, 1)
    query = query.cpu().numpy()


# Measure distance with Gallery and 1 Query
dimension = 128 # image size
gallery_emb = gallery_emb.astype("float32")
query = query.astype("float32")

# IndexFlatL2
index = faiss.IndexFlatL2(dimension)
index.add(gallery_emb)

k = 5
distances, indices = index.search(query, k)
print(f"Indices of {k} of nearest neighbors: {indices}")
print(f"Distances of {k} of nearest neighbors: {distances}")


# Measure distance with Gallery and Queries
correct = 0
with torch.no_grad():
  for i in range(len(val_data)):
    q_input, q_label = val_data[i]
    q_input = q_input.unsqueeze(0).to(device)
    query = best_model.feat_extractor(q_input)
    query = torch.flatten(query, 1)
    query = query.cpu().numpy().astype("float32")

    # IndexFlatL2
    distances, indices = index.search(query, k)
    min_idx = indices[0][0]
    print(f"{i+1} Min Index: {min_idx}, Min Distance: {distances[0][0]:.4f}")
    print(f"Euclidean Distance Pred: {gallery_label[min_idx]}, GT: {q_label}")
    print("------------------------------------------------------------")

    if gallery_label[min_idx] == q_label:
      correct += 1

  accuracy = correct / len(val_data)
  print(f"Accuracy: {accuracy:.4f}")
  
