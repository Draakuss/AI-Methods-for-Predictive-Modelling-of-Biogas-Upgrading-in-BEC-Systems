# AI Models for the Predictive Modelling of Biogas upgrading in BEC's

## Description
This repository contains optimised SVR, RFR, XGBoost and ANN (Using Adams and Levenberg-Marquardt) produced for the purpose of modelling the concentrations of biomethane, acetic acid and hydrogen produced via biogas upgrading in a BEC

These models used four input parameters: Microbial concentration, electrical conductivity, average current and pH

## Features
The Levenberg-Marquardt models contains novel functions to enable the use of the SciPy 'least_squares' function to be usd in a PyTorch ANN environment
