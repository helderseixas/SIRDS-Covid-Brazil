# SIRDS-Covid-Brazil
Repository for Code and Data of the Project Implementing the SIRDS Model for the First Four Covid-19 Waves in Brazil

The results of this project are described in the paper titled "Estimating epidemiological parameters and underreporting of Covid-19 cases in Brazil using a multi-wave mathematical model". If you use this code and data, cite us:

Hélder Lima, Unaí Tupinambás, Frederico Guimarães et al. Estimating epidemiological parameters and underreporting of Covid-19 cases in Brazil using a multi-wave mathematical model, 13 July 2023, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-3161529/v1]

## Reproducibility

To reproduce the results of this paper, please follow the steps below:

1. Prepare Datasets

If you wish to reproduce the process of generating the dataset, please refer to the instructions in the file data/input/README.txt. Alternatively, if you prefer to skip this step, you can proceed to the next item as the dataset is already prepared in the data/output folder.

2. Plot Charts

Execute the notebooks 03_plot_covid_charts, 04_plot_mobility_chart, and 05_plot_vaccination_chart to generate the charts presented in the paper.

3. Optimization Simulations

To run the optimization simulations, execute the notebook 06_optimization_national_sirds.

4. Analyze the Results

To analyze the results and generate the charts presented in the paper, execute the notebook 07_analysis_national_sirds_results.

## Contributors

Hélder Seixas Lima(1,2), Unaí Tupinambás(2), and Frederico Gadelha Guimarães(2)

1) Federal Institute of Northern Minas Gerais (IFNMG)
2) Federal University of Minas Gerais (UFMG)
