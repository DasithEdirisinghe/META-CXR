
# META-CXR: Multimodal Expert Tokens-Based Vision-Language Model for Abnormality-guided Chest X-Ray Reporting

META-CXR is an advanced vision-language framework that bridges the gap between automated chest X-ray (CXR) interpretation and clinical application. Designed to provide accurate abnormality classification and generate coherent radiology reports, META-CXR incorporates multimodal learning, expert token representations, and cutting-edge vision encoders and large languge models(LLM) to enhance diagnostic precision and explainability.

---

<<<<<<< HEAD
**Authors:** [Dasith Edirisinghe][de], [Wimukthi Nimalsiri][wn], [Mahela Hennayake][mh]
=======
**Authors:** [Dasith Edirisinghe][de], [Wimukthi Nimalsiri][wn], [Mahela Hennayake][mh], [Dulani Meedeniya][dm], [ Isabel de la Torre D´ıez][it]
>>>>>>> dev/new-updates

[de]:https://dasithedirisinghe.github.io/
[wn]:https://wimukti.github.io/#/home
[mh]:https://lk.linkedin.com/in/mahela97
<<<<<<< HEAD


**✨ News ✨**
=======
[dm]:https://uom.lk/staff/Meedeniya.DA
[it]:https://scholar.google.es/citations?hl=es&user=82k6rgsAAAAJ


**✨ News ✨**
- 30 Dec 2024: Submitted this work to [Informatics in Medicine Unlocked](https://www.sciencedirect.com/journal/informatics-in-medicine-unlocked)
>>>>>>> dev/new-updates
---


## ✨ Key Features

1. **Multi-Class Abnormality Classification**:
   - META-CXR introduces a classification mechanism that identifies abnormalities as **positive**, **negative**, or **uncertain**, mimicking real-world clinical scenarios.
   - Tailored thresholds for each class improve inference reliability and ensure precise abnormality detection.

2. **Radiology Report Generation**:
   - Utilizes a large language model (LLM) integrated with abnormality findings to generate clinically relevant radiology reports.
   - Reports align with clinical standards, maintaining semantic coherence and diagnostic accuracy.

3. **Expert Token Representations**:
   - Learnable query tokens extract abnormality-specific high-level features, significantly improving classification and report generation tasks.

4. **Multi-Encoder Vision Fusion**:
   - Combines the strengths of ResNet50, Vision Transformer (ViT), and Swin Transformer to capture local and global features.
   - Demonstrates superior performance by leveraging diverse architectural capabilities for feature extraction.

5. **Explainable AI (XAI)**:
   - Includes attention map visualizations that highlight critical regions in the X-rays, offering transparent and interpretable outputs.
   - Enables clinicians to understand the rationale behind model predictions.

6. **Web Interface**:
   - A user-friendly interface allows users to upload chest X-rays for instant classification, report generation, and attention map visualization.
   - Designed for real-time use, making META-CXR accessible to clinicians and researchers alike.

---

<<<<<<< HEAD
## 🌐 Product

=======
## 🌐 Web Interface

META-CXR provides a robust web application **chestxpert.live** that brings its capabilities to clinicians' fingertips.
[Visit the META-CXR Web Interface](https://chestxpert.live)
>>>>>>> dev/new-updates

Key functionalities include:

- **X-Ray Upload**: Easily upload chest X-rays via drag-and-drop or file selection.
- **Classification Results**: View detailed multi-class predictions for abnormalities as positive, negative, or uncertain.
- **Radiology Report Generation**: Obtain automated, clinically relevant reports tailored to the uploaded image.
- **Attention Map Visualization**: Explore attention overlays to understand which regions influenced the model's predictions.

<<<<<<< HEAD
=======
This interface ensures that META-CXR seamlessly integrates into clinical workflows, enhancing accessibility and usability.

>>>>>>> dev/new-updates
---

## 🧪 Results

<<<<<<< HEAD
### Abnoramlity Classification Metrics

#### MIMIC-CXR Dataset

Mean Precision, Recall, F1-Score across all 13 pathologies and No Finding 

=======
### Classification Metrics
>>>>>>> dev/new-updates
| Abnormality                | Precision | Recall | F1-Score |
|---------------------|-----------|--------|----------|
| Average             | 0.87      | 0.78   | 0.73     | 

<<<<<<< HEAD
#### CheXpert Dataset

Zero-shot abnormality classification across five pathologies: Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion. <br>
Comparison is done with SOTA classifier ChexZero 

| Model                | AUC | F1-Score |
|---------------------|-----------|--------|
| ChexZero             | 0.889  | 0.606   |
| META-CXR (Ours)             | 0.824  | 0.699   |
=======
>>>>>>> dev/new-updates

### Report Generation Metrics
| Metric    | META-CXR | 
|-----------|-------------|
<<<<<<< HEAD
| BERTScore | 0.426       | 
| CIDEr     | 0.403       | 
| BLEU-4    | 0.102       | 
| ROUGE-L   | 0.280       | 
| METEOR    | 0.173       |
=======
| BERTScore | 0.416       | 
| CIDEr     | 0.403       | 
| BLEU-4    | 0.098        | 
| ROUGE-L   | 0.280       | 
| METEOR    | 0.149       |
>>>>>>> dev/new-updates

---

## 👨‍⚕️ Clinical Impact

META-CXR enhances radiologist workflows by:
- Automating abnormality detection with high precision and recall.
- Generating clinically accurate and coherent radiology reports.
- Providing interpretable insights through attention map visualizations.

---

## 📖 Acknowledgments

META-CXR leverages the MIMIC-CXR dataset and builds upon advancements in vision-language modeling to provide a state-of-the-art solution for chest X-ray analysis.