# Diffusion-Based Wildfire Spread Prediction

## Overview
Using a **Conditional Diffusion Model** to predict the probabilistic progression of wildfires using GOES-16/17 satellite imagery and NASA FIRMS data.

## Project Structure
- `data/`: Satellite and Fire data (Raw & Processed).
- `src/`: Core logic (Diffusion Model, Data Loaders).
- `models/`: Saved model checkpoints.

## Tech Stack
- PyTorch
- HuggingFace Diffusers
- NASA FIRMS & NOAA GOES Data