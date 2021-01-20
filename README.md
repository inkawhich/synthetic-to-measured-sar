# Bridging a Gap in SAR-ATR: Training on Fully Synthetic and Testing on Measured Data

This code allows for training accurate SAR-ATR classifiers on the SAMPLE dataset, and is released in conjunction with [1]. The gist of the problem is achieving accuracy on measure SAR data when training using synthetically generated SAR data. The experimental methodology follows Experiment 4.1 from [2].

**Prerequisites** 

The only prerequisite required to run the code is to download the SAMPLE dataset released in [2]. It will come as a .zip file called ``SAMPLE_Public_Dist_A.zip`` and you must extract it into the ``synthetic-to-measured-sar/`` directory. This assures that the ``dataset_root`` variable in ``main_experiment41_tester.py`` is set properly. To start running the experiments, open a terminal and run the following command:

```
python -u main_experiment41_tester.py 0.
```

Note, the input argument sets the variable *K* which is defined in [1] and [2].

This code has been verified in an environment with Python 3.6, PyTorch 1.3.1 and NumPy 1.16.4. 


**References**

[1] - Nathan Inkawhich, Matthew Inkawhich, Eric Davis, Uttam Majumder, Erin Tripp, Chris Capraro and Yiran Chen, "Bridging a Gap in SAR-ATR: Training on Fully Synthetic and Testing on Measured Data," Preprint (Under Review), 2020.

[2] - Benjamin Lewis, Theresa Scarnati, Elizabeth Sudkamp, John Nehrbass, Stephen Rosencrantz, Edmund Zelnio, "A SAR dataset for ATR development: the Synthetic and Measured Paired Labeled Experiment (SAMPLE)," Proc. SPIE 10987, Algorithms for Synthetic Aperture Radar Imagery XXVI, 109870H (14 May 2019); https://doi.org/10.1117/12.2523460
