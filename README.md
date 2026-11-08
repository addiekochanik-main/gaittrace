# GaitTrace

**Personalized gait monitoring and change detection using machine learning.**

GaitTrace is an experimental project exploring whether machine learning can learn an individual's walking patterns from accelerometer and gyroscope data and identify deviations from their personal baseline.

## Motivation

Walking patterns can change over time, but those changes aren't always easy to recognize or quantify. GaitTrace aims to explore how motion sensor data could make those changes easier to understand.

## Initial prototype

The first version will:

- Analyze walking data from the UCI Human Activity Recognition Using Smartphones dataset.
- Extract features from accelerometer and gyroscope signals.
- Learn a personalized baseline from one participant's walking data.
- Evaluate the model's response to held-out recordings and simulated deviations.
- Visualize the results.

## Dataset

UCI Human Activity Recognition Using Smartphones:
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The dataset contains smartphone motion recordings from 30 participants performing everyday activities. It is not a clinical or longitudinal gait dataset.

## Status

**Preparation:** Repository structure, dataset, and Python environment.

**Next milestone:** Build and evaluate a personalized gait-baseline prototype.

## Disclaimer

GaitTrace is an experimental research prototype, not a medical device. It does not diagnose conditions, predict falls, or provide treatment recommendations.