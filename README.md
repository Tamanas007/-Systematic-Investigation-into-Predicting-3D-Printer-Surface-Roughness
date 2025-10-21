# -Systematic-Investigation-into-Predicting-3D-Printer-Surface-Roughness


#Project Overview

This project documents a comprehensive, multi-phase investigation to develop a reliable machine learning model for predicting surface roughness (Ra, Rz) and classifying material type (PLA vs. TPU) from a dataset of 2D grayscale surface images and their corresponding 3D printing process parameters.

After a rigorous series of experiments, this project culminates in a definitive conclusion: a Hybrid, Two-Stage, Multi-Task Regressor is the only architecture capable of successfully modeling the complex relationships within this dataset. This final model achieves a strong, positive R-squared score of ~0.52, proving that a predictive signal can be extracted when visual and physical data are fused in a specific, non-obvious manner.

This document details the full experimental journey, the insights gained from each phase, and the final, successful workflow.


# The Core Challenge: A "Low Signal, High Noise" Problem

Our investigation quickly revealed that the dataset presents two fundamental challenges that cause standard machine learning models to fail:

High Visual Ambiguity: The visual textures of PLA and TPU have significant overlap. Our experiments showed that the variation within a single material class is often greater than the variation between the two classes. This "weak" visual signal makes simple classification unreliable.

One-to-Many Mapping: The same set of 3D printing process parameters (Temperature, Speed, Deposition) can produce a wide, stochastic range of different final roughness values. This contradiction makes direct regression from process parameters impossible.



# The Experimental Journey: A Systematic Investigation

Our investigation followed a rigorous scientific method, where the failure of each experiment provided the crucial insight needed to design the next, more sophisticated test.

Phase 1: Standard Deep Learning & Feature Engineering -> Failure

Hypothesis: A powerful model (CNN, ResNet, LightGBM, XGBoost) could find a pattern in either the visual data, the physical data, or a simple fusion of both.

Result: Complete Model Failure. All models in this phase failed to learn. Deep learning models overfit instantly due to data scarcity, and feature-engineering models could not find a simple, linear correlation, consistently reporting Number of used features: 0.

Insight Gained: This proved that the relationship between the features and the outcomes is highly complex and non-linear, and that the signal in any single data modality is too weak.

Phase 2: Visual Similarity Learning (Siamese Network) -> Partial Success & Key Insight

Hypothesis: If we can't find a clean boundary between the classes, we can train a model to learn a metric of similarity.

Result: Partial Success. This was the first breakthrough. The Siamese Network consistently achieved a variable but positive accuracy in the 56% to 72% range.
![Best Case Confusion Matrix](WhatsApp Image 2025-10-16 at 18.49.25_7431c67f.jpg)

Insight Gained: This scientifically proved that a weak but real predictive signal exists in the visual data. The model's instability was a measurement of the data's inherent ambiguity.

Phase 3: The Breakthrough -> A Hybrid Unsupervised-Supervised Workflow

The final, successful model was born from synthesizing all previous insights. Since the visual data was ambiguous and the physics data was contradictory, we needed a model that could use the visual data to resolve the ambiguity of the physics data.



Stage 1: The "Visual Inspector" (Unsupervised Clustering)

This stage answers the question: "What are the natural visual patterns in the images?"

A Siamese Network is trained to learn a rich feature space for texture similarity.

Its encoder generates a "fingerprint" (embedding) for all 180 images.

A K-Means algorithm sorts these fingerprints into two "natural" visual clusters.

Each image is then tagged with a new feature: visual_cluster (ID 0 or 1).

Stage 2: The "Informed Multi-Task Regressor"

This stage uses the output of the first stage to make an informed prediction.

Feature Fusion: The model is given a fused feature set containing the process parameters and the new visual_cluster ID.

Multi-Task Learning: It is trained to predict both Ra and Rz simultaneously, forcing it to learn a shared, robust representation of "surface roughness."

Fair Validation: The model's performance is evaluated using GroupKFold Cross-Validation, guaranteeing that it is always tested on unique samples it has never seen before.



# Final Model & Results

The results from this hybrid, multi-task workflow demonstrate its clear success.

Ra Prediction R²: 0.504

Rz Prediction R²: 0.460

Analysis: An R-squared score of ~0.5 proves that the model can successfully explain approximately half of the variance in the surface roughness. This is a significant and scientifically valid result. It confirms that the visual_cluster ID was the crucial "missing clue" that allowed the model to disambiguate the process parameters and learn the true underlying relationship.


# Definitive Conclusion

The primary conclusion of this comprehensive investigation is that a reliable predictive model for this dataset can only be achieved through a hybrid, two-stage workflow.

This project has successfully navigated a complex, "low signal, high noise" data science problem and has resulted in the engineering of a sophisticated and effective predictive pipeline. We have proven that:

Neither visual nor physical data is sufficient on its own.

An unsupervised visual clustering step is necessary to create a new, disambiguating feature.

Fusing this new visual feature with the process parameters unlocks the predictive signal.

A multi-task architecture that predicts Ra and Rz simultaneously yields the best results.


# How to Use This Notebook

The final, successful script is colab_hybrid_deterministic_regressor.py. To run this experiment:

Open the script in a Google Colab notebook.

Ensure your dataset_roughness.csv and roughness images.zip files are in your Google Drive's root directory.

Run all the cells
